using System;
using System.Diagnostics;
using System.IO;
using NumSharp.Examples.MaterialLab.App.Native;
using NumSharp.Examples.MaterialLab.App.Rendering;
using NumSharp.Examples.MaterialLab.Simulation;

namespace NumSharp.Examples.MaterialLab.App
{
    /// <summary>Startup options parsed from the command line.</summary>
    internal sealed class LabOptions
    {
        /// <summary>Start in borderless fullscreen instead of the 2/3-of-the-screen window.</summary>
        public bool Fullscreen;
        /// <summary>Fraction of the monitor the window's client area covers (2/3 by default).</summary>
        public double ScreenFraction = 2.0 / 3.0;
        /// <summary>Initial scene index (0-based).</summary>
        public int Scene;
        /// <summary>Grid rows (resolution of the simulation).</summary>
        public int GridRows = 108;
        /// <summary>Vertical sync (adaptive when supported).</summary>
        public bool VSync = true;
        /// <summary>When set: render this many frames, save the final image to <see cref="ScreenshotPath"/>, exit.</summary>
        public int Frames;
        /// <summary>Screenshot destination for the automated capture mode.</summary>
        public string ScreenshotPath;
        /// <summary>Automated runs: create the window without showing it.</summary>
        public bool Hidden;
        /// <summary>Benchmark mode: print frame statistics after <see cref="Frames"/> frames and exit.</summary>
        public bool Bench;
        /// <summary>Hide the help hint and panels (clean screenshots).</summary>
        public bool NoHud;
        /// <summary>
        /// Let the app lower the grid resolution when a freshly loaded scene cannot hold 60 FPS on this machine.
        /// Off when <c>--quality</c> is given explicitly and in automated runs (screenshots, benchmarks), whose
        /// output must not depend on how busy the machine happened to be.
        /// </summary>
        public bool AutoQuality = true;
    }

    /// <summary>
    /// The Material Lab application: owns the window, the simulation, the renderer and the HUD, and runs the
    /// frame loop — pump input → act on it → advance the simulation one frame → render → present.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Honest FPS.</b> The counter is frames presented per second measured over the whole loop (simulation +
    /// rendering + presentation), not a render-only rate: the simulation runs on the same thread, once per
    /// presented frame, so a frame's number is exactly how long the user waits for it. The HUD breaks the time
    /// down per simulation phase so the cost of each NumSharp stage is visible live.
    /// </para>
    /// <para>
    /// <b>Letterboxing.</b> The tank keeps its 16:9 aspect in any window shape; the rendered image fills the
    /// largest centered 16:9 rectangle and the mouse is mapped through the same rectangle.
    /// </para>
    /// </remarks>
    internal sealed unsafe class LabApp : IDisposable
    {
        /// <summary>Tank aspect ratio (width/height); the image is letterboxed to it in any window shape.</summary>
        private const float Aspect = 16f / 9f;
        /// <summary>Grid rows per quality level. Cost grows roughly with rows³ (cells × particles per cell × substeps).</summary>
        private static readonly int[] QualityRows = { 72, 90, 108, 126 };
        /// <summary>Display names of <see cref="QualityRows"/>.</summary>
        private static readonly string[] QualityNames = { "Low", "Medium", "High", "Ultra" };

        /// <summary>The startup options (automation flags are consulted every frame).</summary>
        private readonly LabOptions _opt;
        /// <summary>The window, its GL context and the input snapshot.</summary>
        private readonly GlWindow _win;
        /// <summary>Fonts for the HUD.</summary>
        private readonly FontAtlas _atlas;
        /// <summary>The 2-D batcher the HUD draws with.</summary>
        private readonly Overlay _overlay;
        /// <summary>Draws the simulation (fields, grains, walls, bloom, tone map).</summary>
        private readonly SceneRenderer _renderer;
        /// <summary>The heads-up display.</summary>
        private readonly Hud _hud;
        /// <summary>The simulation. Replaced wholesale when the quality (grid resolution) changes.</summary>
        private MpmWorld _world;
        /// <summary>Current index into <see cref="QualityRows"/>.</summary>
        private int _quality;

        /// <summary>The scene last loaded (R reloads it; a quality change rebuilds it).</summary>
        private int _sceneIndex;
        /// <summary>The material the brush and faucets place.</summary>
        private MaterialId _material = MaterialId.Water;
        /// <summary>Brush radius in simulation units (the tank is 1 unit tall).</summary>
        private float _brushRadius = 0.045f;
        /// <summary>Simulation paused (rendering continues); help sheet shown.</summary>
        private bool _paused, _showHelp;
        /// <summary>
        /// Cursor position last frame in simulation units (NaN before the first sample) — the mouse velocity that
        /// flung material and the stir inherit, and the start of each painted wall segment.
        /// </summary>
        private float _prevMouseX = float.NaN, _prevMouseY = float.NaN;
        /// <summary>The letterbox rectangle in window pixels (origin top-left): the 16:9 region the image fills.</summary>
        private int _vx0, _vy0, _vw, _vh;   // letterbox rectangle in window pixels (origin top-left)

        /// <summary>Wall clock since startup (toasts, shader time).</summary>
        private readonly Stopwatch _clock = Stopwatch.StartNew();
        /// <summary>FPS, the frame-time history graph and the benchmark sums.</summary>
        private readonly FrameMeter _meter = new FrameMeter();
        /// <summary>The transient message currently shown.</summary>
        private string _toast = "";
        /// <summary>Clock time (s) when <see cref="_toast"/> disappears.</summary>
        private double _toastUntil;

        // Auto quality (see LabOptions.AutoQuality): frames observed since the last scene load, and the summed
        // simulation + render CPU time of the measured window.
        /// <summary>Auto quality still active (off after Q, with --quality, and in automated runs).</summary>
        private bool _autoQuality;
        /// <summary>Frames observed since the last scene load (parked far negative once a decision is made).</summary>
        private int _autoFrames;
        /// <summary>Summed simulation + render CPU milliseconds over the measured window.</summary>
        private double _autoCostMs;

        /// <summary>Frames after a scene load that auto quality ignores (JIT warm-up, the scene's first impact).</summary>
        private const int AutoSkipFrames = 40;
        /// <summary>Frames auto quality averages over before deciding (~1.5 s at 60 FPS).</summary>
        private const int AutoMeasureFrames = 90;
        /// <summary>
        /// CPU milliseconds per frame (simulation + render submit) that still leave a 60 FPS frame (16.7 ms) room
        /// for the GPU and presentation.
        /// </summary>
        private const double AutoBudgetMs = 14.5;

        /// <summary>Creates the window, the GL pipeline and the first scene.</summary>
        /// <param name="options">Startup options.</param>
        public LabApp(LabOptions options)
        {
            _opt = options;
            _win = GlWindow.Create("NumSharp Material Lab", options.ScreenFraction, options.Fullscreen, options.Hidden);
            _win.SetVSync(options.VSync && !options.Bench);
            _atlas = new FontAtlas();
            _overlay = new Overlay(_atlas);
            UpdateViewport();
            _renderer = new SceneRenderer(_vw, _vh);
            _hud = new Hud(_overlay, _atlas);
            _quality = Math.Max(0, Array.IndexOf(QualityRows, options.GridRows));
            if (Array.IndexOf(QualityRows, options.GridRows) < 0) _quality = 2;
            _autoQuality = options.AutoQuality && options.Frames == 0;
            _world = new MpmWorld(options.GridRows, Aspect, seed: 7);
            _sceneIndex = Math.Clamp(options.Scene, 0, Scenes.All.Length - 1);
            LoadScene(_sceneIndex);
        }

        /// <summary>Runs until the window closes (or the automated frame budget is spent).</summary>
        /// <returns>Process exit code.</returns>
        public int Run()
        {
            int frame = 0;
            double lastTitle = 0;
            while (!_win.CloseRequested)
            {
                long t0 = Stopwatch.GetTimestamp();
                _win.PumpMessages();
                if (_win.CloseRequested) break;
                if (_win.Resized) { _win.Resized = false; UpdateViewport(); _renderer.Resize(_vw, _vh); }

                HandleInput();

                long t1 = Stopwatch.GetTimestamp();
                if (!_paused) _world.StepFrame();
                long t2 = Stopwatch.GetTimestamp();

                var brush = BrushCursorFor();
                _renderer.Render(_world, brush, (float)_clock.Elapsed.TotalSeconds);
                if (!_opt.NoHud)
                {
                    _overlay.Begin(_renderer.Width, _renderer.Height);
                    _hud.Draw(new HudState
                    {
                        Width = _renderer.Width, Height = _renderer.Height,
                        Fps = _meter.Fps, FrameMs = _meter.LastFrameMs, SimMs = _paused ? 0 : _world.Stats.TotalMs,
                        RenderMs = _meter.LastRenderMs, Stats = _world.Stats, History = _meter.History, HistoryHead = _meter.Head,
                        Paused = _paused, ShowHelp = _showHelp, Material = _material, Scene = Scenes.All[_sceneIndex],
                        SceneIndex = _sceneIndex, Transfers = _world.Transfers, GridNx = _world.Grid.Nx, GridNy = _world.Grid.Ny,
                        Quality = QualityNames[_quality], VSync = _win.SwapInterval != 0, GravityAngle = _world.GravityAngle,
                        Toast = _clock.Elapsed.TotalSeconds < _toastUntil ? _toast : null, World = _world,
                        Faucets = _world.Emitters.Count, GpuName = _win.GlRenderer,
                    });
                    _overlay.End();
                }
                long t3 = Stopwatch.GetTimestamp();
                if (_autoQuality && !_paused) ObserveAutoQuality((t3 - t1) * 1000.0 / Stopwatch.Frequency);

                frame++;
                if (_opt.Frames > 0 && frame >= _opt.Frames)
                {
                    if (_opt.ScreenshotPath != null) SaveScreenshot(_opt.ScreenshotPath);
                    if (_opt.Bench) Console.WriteLine(_meter.Report(_world));
                    break;
                }

                // Letterbox bars, then the frame.
                Gl.BindFramebuffer(Gl.FRAMEBUFFER, 0);
                Gl.Viewport(0, 0, _win.Width, _win.Height);
                Gl.ClearColor(0, 0, 0, 1);
                Gl.Clear(Gl.COLOR_BUFFER_BIT);
                PresentLetterboxed();
                _win.Swap();

                long t4 = Stopwatch.GetTimestamp();
                _meter.Record((t4 - t0) * 1000.0 / Stopwatch.Frequency, (t3 - t2) * 1000.0 / Stopwatch.Frequency, (t2 - t1) * 1000.0 / Stopwatch.Frequency,
                    _paused ? default : _world.Stats);
                if (_clock.Elapsed.TotalSeconds - lastTitle > 0.5)
                {
                    lastTitle = _clock.Elapsed.TotalSeconds;
                    _win.SetTitle($"NumSharp Material Lab — {_meter.Fps:F0} FPS — {Scenes.All[_sceneIndex].Name} — {_world.ParticleCount:N0} particles");
                }
                _win.Input.EndFrame();
            }
            return 0;
        }

        /// <summary>Draws the final image into the letterbox rectangle of the window.</summary>
        private void PresentLetterboxed()
        {
            // GL's viewport origin is bottom-left; our rectangle is stored top-left.
            int glY = _win.Height - (_vy0 + _vh);
            Gl.Viewport(_vx0, glY, _vw, _vh);
            _renderer.Present(_vw, _vh);
            Gl.Viewport(0, 0, _win.Width, _win.Height);
        }

        /// <summary>Recomputes the largest centered 16:9 rectangle in the window.</summary>
        private void UpdateViewport()
        {
            int w = _win.Width, h = _win.Height;
            int vw = w, vh = (int)MathF.Round(w / Aspect);
            if (vh > h) { vh = h; vw = (int)MathF.Round(h * Aspect); }
            _vw = Math.Max(16, vw); _vh = Math.Max(16, vh);
            _vx0 = (w - _vw) / 2; _vy0 = (h - _vh) / 2;
        }

        /// <summary>Maps the cursor to simulation units (y up), or false when it is outside the tank image.</summary>
        private bool MouseToWorld(out float x, out float y)
        {
            var inp = _win.Input;
            float u = (inp.MouseX - _vx0) / (float)_vw, v = (inp.MouseY - _vy0) / (float)_vh;
            x = u * _world.Grid.Width;
            y = (1f - v) * 1f;
            return u >= 0 && u <= 1 && v >= 0 && v <= 1;
        }

        /// <summary>The brush ring for the renderer (color of the tool/material).</summary>
        private BrushCursor BrushCursorFor()
        {
            bool inside = MouseToWorld(out float x, out float y);
            var inp = _win.Input;
            (float, float, float) color = Materials.Get(_material).Color;
            if (inp.Shift) color = (0.8f, 0.8f, 0.85f);
            else if (inp.RightDown) color = (0.35f, 0.75f, 1.0f);
            else if (inp.MiddleDown || inp.Ctrl) color = (1.0f, 0.35f, 0.3f);
            return new BrushCursor(x, y, _brushRadius, inside && inp.MouseSeen && !_opt.NoHud, color);
        }

        /// <summary>Applies this frame's keyboard and mouse input.</summary>
        private void HandleInput()
        {
            var inp = _win.Input;
            foreach (var key in inp.Presses)
                OnKey(key.VirtualKey, key.IsRepeat);

            if (inp.WheelNotches != 0)
                _brushRadius = Math.Clamp(_brushRadius * MathF.Pow(1.15f, inp.WheelNotches), 0.012f, 0.25f);

            // Gravity tilt while an arrow key is held (the whole tank "tips").
            float tilt = (inp.KeyDown[0x27] ? 1f : 0f) - (inp.KeyDown[0x25] ? 1f : 0f);   // VK_RIGHT / VK_LEFT
            if (tilt != 0) _world.GravityAngle = Math.Clamp(_world.GravityAngle + tilt * 0.035f, -1.4f, 1.4f);

            bool inside = MouseToWorld(out float mx, out float my);
            float vx = 0, vy = 0;
            float pmx = float.IsNaN(_prevMouseX) ? mx : _prevMouseX, pmy = float.IsNaN(_prevMouseY) ? my : _prevMouseY;
            if (!float.IsNaN(_prevMouseX))
            {
                // Mouse speed in simulation units per simulated second (one frame advances FrameTime).
                vx = Math.Clamp((mx - _prevMouseX) / _world.FrameTime, -25f, 25f);
                vy = Math.Clamp((my - _prevMouseY) / _world.FrameTime, -25f, 25f);
            }
            _prevMouseX = mx; _prevMouseY = my;

            bool stirring = inp.RightDown && !inp.Shift && inside;
            _world.GridStep.SetStir(mx, my, vx, vy, _brushRadius * 1.6f, stirring ? 0.35f : 0f);
            if (!inside) return;

            // Walls are painted as capsules from the previous cursor sample to this one, so a fast stroke is one
            // continuous smooth wall rather than a row of dots.
            if (inp.Shift && inp.LeftDown) _world.Grid.PaintSegment(pmx, pmy, mx, my, _brushRadius * 0.45f, solid: true);
            else if (inp.Shift && inp.RightDown) _world.Grid.PaintSegment(pmx, pmy, mx, my, _brushRadius * 0.9f, solid: false);
            else if (inp.MiddleDown || (inp.Ctrl && inp.LeftDown)) _world.Erase(mx, my, _brushRadius);
            else if (inp.LeftDown) _world.Paint(_material, mx, my, _brushRadius, vx * 0.6f, vy * 0.6f);
        }

        /// <summary>Handles one key press.</summary>
        /// <param name="vk">Virtual key.</param>
        /// <param name="repeat">Auto-repeat flag.</param>
        private void OnKey(int vk, bool repeat)
        {
            // Material keys 1..9, 0.
            if (vk >= 0x31 && vk <= 0x39) { SelectMaterial((MaterialId)(vk - 0x31)); return; }
            if (vk == 0x30) { SelectMaterial(MaterialId.Rock); return; }
            // Scenes F1..F10.
            if (vk >= 0x70 && vk <= 0x79 && !repeat) { LoadScene(vk - 0x70); return; }
            switch (vk)
            {
                case 0x1B: _win.RequestClose(); return;                                         // Esc
                case 0x7A: if (!repeat) { _win.ToggleFullscreen(); } return;                    // F11
                case 0x7B: if (!repeat) SaveScreenshot(null); return;                           // F12
                case 0x20: if (!repeat) { _paused = !_paused; Toast(_paused ? "Paused" : "Running"); } return;   // Space
                case 0x52: if (!repeat) LoadScene(_sceneIndex); return;                         // R
                case 0x43: if (!repeat) { _world.ClearParticles(); Toast("Cleared particles"); } return;         // C
                case 0x47: if (!repeat) { _world.GravityAngle = 0; Toast("Gravity reset"); } return;             // G
                case 0x48: if (!repeat) _showHelp = !_showHelp; return;                         // H
                case 0x56: if (!repeat) { _win.SetVSync(_win.SwapInterval == 0); Toast(_win.SwapInterval == 0 ? "VSync off" : "VSync on"); } return;   // V
                case 0x54:                                                                       // T
                    if (!repeat)
                    {
                        _world.Transfers = _world.Transfers == TransferMode.Fused ? TransferMode.Reference : TransferMode.Fused;
                        Toast(_world.Transfers == TransferMode.Fused ? "Transfers: fused SIMD kernels" : "Transfers: pure NumSharp (np.bincount / np.take)");
                    }
                    return;
                case 0x51: if (!repeat) CycleQuality(); return;                                 // Q
                case 0x46:                                                                       // F: faucet
                    if (repeat) return;
                    if (_win.Input.Shift) { _world.Emitters.Clear(); Toast("Faucets removed"); }
                    else if (MouseToWorld(out float fx, out float fy))
                    {
                        _world.Emitters.Add(new Emitter { Material = _material, X = fx, Y = fy, Radius = Math.Max(0.02f, _brushRadius * 0.6f), VY = -1.5f });
                        Toast($"{Materials.Get(_material).Name} faucet placed");
                    }
                    return;
                case 0xBB: case 0x6B: _brushRadius = Math.Min(0.25f, _brushRadius * 1.15f); return;    // + / numpad +
                case 0xBD: case 0x6D: _brushRadius = Math.Max(0.012f, _brushRadius / 1.15f); return;   // - / numpad -
            }
        }

        /// <summary>Selects the painting material.</summary>
        private void SelectMaterial(MaterialId id)
        {
            _material = id;
            Toast($"{Materials.Get(id).Name} selected");
        }

        /// <summary>Loads a scene by index (also restarts the current one) and restarts the auto-quality measurement.</summary>
        /// <param name="index">Scene index (clamped).</param>
        private void LoadScene(int index)
        {
            _sceneIndex = Math.Clamp(index, 0, Scenes.All.Length - 1);
            var s = Scenes.Load(_world, _sceneIndex);
            _autoFrames = 0;
            _autoCostMs = 0;
            Toast($"F{_sceneIndex + 1}  {s.Name}");
        }

        /// <summary>
        /// Auto quality: after a scene loads, skips the warm-up frames, averages the CPU cost of the next
        /// <see cref="AutoMeasureFrames"/>, and if that does not fit a 60 FPS frame, drops the grid one level (the
        /// scene restarts at the coarser resolution and is measured again). It only ever steps DOWN, and it stops
        /// for good once the user picks a quality with Q — a choice the user made is never overridden.
        /// </summary>
        /// <param name="frameCostMs">This frame's simulation + render CPU time.</param>
        private void ObserveAutoQuality(double frameCostMs)
        {
            _autoFrames++;
            if (_autoFrames <= AutoSkipFrames) return;
            _autoCostMs += frameCostMs;
            if (_autoFrames < AutoSkipFrames + AutoMeasureFrames) return;
            double average = _autoCostMs / AutoMeasureFrames;
            _autoFrames = int.MinValue / 2;   // measured: wait for the next scene load
            if (average > AutoBudgetMs && _quality > 0)
            {
                SetQuality(_quality - 1);
                Toast($"Auto quality: {QualityNames[_quality]} (grid {_world.Grid.Nx}×{_world.Grid.Ny}) to hold 60 FPS — Q to change");
            }
        }

        /// <summary>Rebuilds the world at the next grid resolution and reloads the scene (the user's choice: auto quality stops).</summary>
        private void CycleQuality()
        {
            _autoQuality = false;
            SetQuality((_quality + 1) % QualityRows.Length);
            Toast($"Quality {QualityNames[_quality]} — grid {_world.Grid.Nx}×{_world.Grid.Ny}");
        }

        /// <summary>
        /// Replaces the world with one at quality level <paramref name="quality"/> (same transfer mode) and reloads
        /// the current scene into it. Painted walls and hand-placed material are lost — they belong to a grid.
        /// </summary>
        /// <param name="quality">Index into the quality table.</param>
        private void SetQuality(int quality)
        {
            _quality = quality;
            var old = _world;
            var transfers = old.Transfers;
            _world = new MpmWorld(QualityRows[_quality], Aspect, seed: 7) { Transfers = transfers };
            old.Dispose();
            LoadScene(_sceneIndex);
        }

        /// <summary>Shows a transient message for two seconds.</summary>
        private void Toast(string message)
        {
            _toast = message;
            _toastUntil = _clock.Elapsed.TotalSeconds + 2.0;
        }

        /// <summary>Saves the final image (with HUD) as PNG.</summary>
        /// <param name="path">File path, or null for a timestamped name in the current directory.</param>
        private void SaveScreenshot(string path)
        {
            path ??= Path.Combine(Environment.CurrentDirectory, $"MaterialLab_{DateTime.Now:yyyyMMdd_HHmmss}.png");
            var rgb = _renderer.ReadFinalRgb();
            PngWriter.Write(path, rgb, _renderer.Width, _renderer.Height);
            Toast($"Saved {Path.GetFileName(path)}");
            Console.WriteLine($"screenshot: {path}");
        }

        /// <summary>Releases everything.</summary>
        public void Dispose()
        {
            _renderer.Dispose();
            _overlay.Dispose();
            _atlas.Dispose();
            _world.Dispose();
            _win.Dispose();
        }
    }

    /// <summary>
    /// Frame-time bookkeeping: a ring of recent frame durations (for the HUD graph), an FPS readout averaged
    /// over the last half second, and the benchmark summary.
    /// </summary>
    internal sealed class FrameMeter
    {
        /// <summary>Ring buffer of frame times (ms).</summary>
        public readonly float[] History = new float[180];
        /// <summary>Next write index in <see cref="History"/>.</summary>
        public int Head;
        /// <summary>Frames per second over the last ~half second.</summary>
        public double Fps { get; private set; }
        /// <summary>Most recent frame time (ms).</summary>
        public double LastFrameMs { get; private set; }
        /// <summary>Most recent render (GPU submit + HUD) time on the CPU (ms).</summary>
        public double LastRenderMs { get; private set; }

        /// <summary>Milliseconds accumulated in the current half-second FPS window.</summary>
        private double _windowMs;
        /// <summary>Frames in the current FPS window.</summary>
        private int _windowFrames;
        /// <summary>Benchmark sums: whole frame, simulation, render (ms).</summary>
        private double _sumFrame, _sumSim, _sumRender;
        /// <summary>Benchmark sums of the world's per-phase timings (ms) and substep counts.</summary>
        private double _sumConstitutive, _sumP2G, _sumGrid, _sumG2P, _sumOther, _sumSubsteps;
        /// <summary>Frames summed into the benchmark totals.</summary>
        private int _samples;

        /// <summary>Records one frame.</summary>
        /// <param name="frameMs">Whole-loop time.</param>
        /// <param name="renderMs">Render + HUD CPU time.</param>
        /// <param name="simMs">Simulation time.</param>
        /// <param name="phases">The world's per-phase timing for this frame (default when paused) — summed for the benchmark breakdown.</param>
        public void Record(double frameMs, double renderMs, double simMs, in FrameStats phases)
        {
            _sumConstitutive += phases.ConstitutiveMs; _sumP2G += phases.P2GMs; _sumGrid += phases.GridMs;
            _sumG2P += phases.G2PMs; _sumOther += phases.OtherMs; _sumSubsteps += phases.Substeps;
            LastFrameMs = frameMs;
            LastRenderMs = renderMs;
            History[Head] = (float)frameMs;
            Head = (Head + 1) % History.Length;
            _windowMs += frameMs;
            _windowFrames++;
            if (_windowMs >= 500)
            {
                Fps = _windowFrames * 1000.0 / _windowMs;
                _windowMs = 0;
                _windowFrames = 0;
            }
            _sumFrame += frameMs; _sumSim += simMs; _sumRender += renderMs; _samples++;
        }

        /// <summary>A one-line summary for benchmark runs.</summary>
        /// <param name="world">The world (for particle count).</param>
        /// <returns>The report.</returns>
        public string Report(MpmWorld world)
        {
            if (_samples == 0) return "no frames";
            double n = _samples;
            return $"{_samples} frames: avg frame {_sumFrame / n:F2} ms ({1000.0 * n / _sumFrame:F1} fps), sim {_sumSim / n:F2} ms, render {_sumRender / n:F2} ms, {world.ParticleCount} particles"
                 + $" | per frame: constitutive {_sumConstitutive / n:F2}, P2G {_sumP2G / n:F2}, grid {_sumGrid / n:F2}, G2P {_sumG2P / n:F2}, other {_sumOther / n:F2} ms over {_sumSubsteps / n:F1} substeps";
        }
    }
}
