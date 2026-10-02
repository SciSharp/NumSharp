using System;
using NumSharp.Examples.MaterialLab.App.Rendering;
using NumSharp.Examples.MaterialLab.Simulation;

namespace NumSharp.Examples.MaterialLab.App
{
    /// <summary>Everything the HUD shows for one frame (a snapshot, so drawing never touches live state twice).</summary>
    internal struct HudState
    {
        /// <summary>Image width.</summary>
        public int Width;
        /// <summary>Image height.</summary>
        public int Height;
        /// <summary>Frames per second (whole loop).</summary>
        public double Fps;
        /// <summary>Last frame time (ms).</summary>
        public double FrameMs;
        /// <summary>Simulation time of the last frame (ms).</summary>
        public double SimMs;
        /// <summary>Render + HUD CPU time of the last frame (ms).</summary>
        public double RenderMs;
        /// <summary>Solver statistics.</summary>
        public FrameStats Stats;
        /// <summary>Frame-time ring buffer.</summary>
        public float[] History;
        /// <summary>Ring write index.</summary>
        public int HistoryHead;
        /// <summary>Simulation paused.</summary>
        public bool Paused;
        /// <summary>Help overlay visible.</summary>
        public bool ShowHelp;
        /// <summary>Selected material.</summary>
        public MaterialId Material;
        /// <summary>Current scene.</summary>
        public Scenes.Scene Scene;
        /// <summary>Current scene index.</summary>
        public int SceneIndex;
        /// <summary>Transfer implementation in use.</summary>
        public TransferMode Transfers;
        /// <summary>Grid columns.</summary>
        public int GridNx;
        /// <summary>Grid rows.</summary>
        public int GridNy;
        /// <summary>Quality preset name.</summary>
        public string Quality;
        /// <summary>Vsync on.</summary>
        public bool VSync;
        /// <summary>Gravity tilt (radians).</summary>
        public float GravityAngle;
        /// <summary>Transient message, or null.</summary>
        public string Toast;
        /// <summary>The world (particle counts per material).</summary>
        public MpmWorld World;
        /// <summary>Faucets running.</summary>
        public int Faucets;
        /// <summary>GPU name.</summary>
        public string GpuName;
    }

    /// <summary>
    /// Draws the heads-up display with the <see cref="Overlay"/>: the performance panel (big FPS counter,
    /// frame-time graph, per-phase NumSharp timing), the scene title, the material palette with live counts,
    /// transient toasts, and the help sheet.
    /// </summary>
    internal sealed class Hud
    {
        /// <summary>The batcher every HUD element is drawn through.</summary>
        private readonly Overlay _o;
        /// <summary>The baked fonts (faces for UI text, headings, numbers and the big FPS digits).</summary>
        private readonly FontAtlas _f;

        /// <summary>Panel background: near-black at 62 % so the simulation stays visible behind every panel.</summary>
        private static readonly Rgba Panel = new Rgba(0.02f, 0.025f, 0.035f, 0.62f);
        /// <summary>Primary text color.</summary>
        private static readonly Rgba Text = new Rgba(0.90f, 0.92f, 0.96f);
        /// <summary>Secondary text (units, hints, inactive entries).</summary>
        private static readonly Rgba Dim = new Rgba(0.62f, 0.66f, 0.74f);
        /// <summary>Accent for key hints and the NumSharp timing line.</summary>
        private static readonly Rgba Accent = new Rgba(0.45f, 0.78f, 1.0f);

        /// <summary>Creates the HUD over an overlay and its atlas.</summary>
        /// <param name="overlay">The overlay batcher.</param>
        /// <param name="atlas">The font atlas.</param>
        public Hud(Overlay overlay, FontAtlas atlas)
        {
            _o = overlay;
            _f = atlas;
        }

        /// <summary>Queues the whole HUD for this frame (between <see cref="Overlay.Begin"/> and <see cref="Overlay.End"/>).</summary>
        /// <param name="s">The frame's state.</param>
        public void Draw(in HudState s)
        {
            DrawPerformance(s);
            DrawSceneTitle(s);
            DrawPalette(s);
            if (s.Paused) Centered(_f.Big, "PAUSED", s.Height * 0.40f, new Rgba(1f, 1f, 1f, 0.85f), s.Width);
            if (s.Toast != null) Centered(_f.UiBold, s.Toast, 64, new Rgba(1f, 0.95f, 0.8f), s.Width);
            if (s.ShowHelp) DrawHelp(s);
        }

        /// <summary>The performance panel (top-left).</summary>
        private void DrawPerformance(in HudState s)
        {
            float x = 14, y = 14, w = 348, h = 188;
            _o.Rect(x, y, w, h, Panel);
            _o.ShadowText(_f.UiBold, "NumSharp Material Lab", x + 12, y + 8, Text);

            var fpsColor = s.Fps >= 55 ? new Rgba(0.45f, 1.0f, 0.55f) : s.Fps >= 30 ? new Rgba(1.0f, 0.85f, 0.35f) : new Rgba(1.0f, 0.42f, 0.35f);
            float adv = _o.ShadowText(_f.Big, s.Fps.ToString("F0"), x + 12, y + 30, fpsColor);
            _o.ShadowText(_f.UiBold, "FPS", x + 18 + adv, y + 54, fpsColor);
            _o.ShadowText(_f.Mono, $"{s.FrameMs,5:F1} ms/frame", x + 200, y + 40, Text);
            _o.ShadowText(_f.Mono, s.VSync ? "vsync on" : "vsync off", x + 200, y + 58, Dim);

            // Frame-time graph: last 180 frames, full scale 33 ms, guide lines at 16.7 ms (60 fps) and 33 ms.
            float gx = x + 12, gy = y + 84, gw = w - 24, gh = 34;
            _o.Rect(gx, gy, gw, gh, new Rgba(1, 1, 1, 0.04f));
            _o.Rect(gx, gy + gh * (1 - 16.7f / 33.3f), gw, 1, new Rgba(0.45f, 1.0f, 0.55f, 0.35f));
            int n = s.History.Length;
            float bw = gw / n;
            for (int i = 0; i < n; i++)
            {
                float ms = s.History[(s.HistoryHead + i) % n];
                if (ms <= 0) continue;
                float bh = Math.Min(gh, gh * ms / 33.3f);
                var c = ms <= 17.5f ? new Rgba(0.40f, 0.85f, 0.55f, 0.85f) : ms <= 34f ? new Rgba(1.0f, 0.80f, 0.30f, 0.9f) : new Rgba(1.0f, 0.38f, 0.32f, 0.95f);
                _o.Rect(gx + i * bw, gy + gh - bh, Math.Max(1, bw - 0.2f), bh, c);
            }

            var st = s.Stats;
            float ly = gy + gh + 6;
            _o.ShadowText(_f.Mono, $"sim {s.SimMs,5:F1} ms   render {s.RenderMs,4:F1} ms", gx, ly, Text);
            _o.ShadowText(_f.Mono, $"{st.Particles,7:N0} particles  {st.Substeps}×{st.Dt * 1e6f,4:F0} µs substeps", gx, ly + 16, Text);
            _o.ShadowText(_f.Mono, $"NumSharp stress {st.ConstitutiveMs,4:F1} · grid {st.GridMs,4:F1} ms", gx, ly + 32, Accent);
            string tr = s.Transfers == TransferMode.Fused ? "fused SIMD" : "np.bincount/np.take";
            _o.ShadowText(_f.Mono, $"P2G {st.P2GMs,4:F1} · G2P {st.G2PMs,4:F1} ms  [T] {tr}", gx, ly + 48, Dim);
        }

        /// <summary>Scene name, description and navigation hint (top-right).</summary>
        private void DrawSceneTitle(in HudState s)
        {
            string title = $"F{s.SceneIndex + 1}  {s.Scene.Name}";
            float tw = Math.Max(_f.UiBold.Measure(title), _f.Ui.Measure(s.Scene.Description));
            float w = tw + 28, x = s.Width - w - 14, y = 14;
            _o.Rect(x, y, w, 70, Panel);
            _o.ShadowText(_f.UiBold, title, x + 14, y + 8, Text);
            _o.ShadowText(_f.Ui, s.Scene.Description, x + 14, y + 28, Dim);
            string info = $"grid {s.GridNx}×{s.GridNy} · {s.Quality} [Q] · H help";
            if (s.GravityAngle != 0) info = $"gravity tilted {s.GravityAngle * 180f / MathF.PI:+0;-0}° [G] · " + info;
            if (s.Faucets > 0) info = $"{s.Faucets} faucet{(s.Faucets > 1 ? "s" : "")} · " + info;
            _o.ShadowText(_f.Ui, info, x + 14, y + 47, Accent);
        }

        /// <summary>
        /// Material palette (bottom): one slim row of chips — swatch, key and name, plus the live particle count
        /// of materials present. Kept low and translucent because the bottom of the tank is where material rests.
        /// </summary>
        private void DrawPalette(in HudState s)
        {
            int count = Materials.Count;
            float cellW = Math.Min(118, (s.Width - 40) / (float)count), cellH = 24;
            float total = cellW * count, x0 = (s.Width - total) / 2, y0 = s.Height - cellH - 6;
            _o.Rect(x0 - 6, y0 - 3, total + 12, cellH + 6, new Rgba(0.02f, 0.025f, 0.035f, 0.50f));
            for (int i = 0; i < count; i++)
            {
                var m = Materials.All[i];
                float cx = x0 + i * cellW;
                bool sel = m.Id == s.Material;
                if (sel)
                {
                    _o.Rect(cx + 1, y0, cellW - 2, cellH, new Rgba(1, 1, 1, 0.12f));
                    _o.Frame(cx + 1, y0, cellW - 2, cellH, new Rgba(1, 1, 1, 0.85f), 1.5f);
                }
                var c = new Rgba(Srgb(m.Color.R), Srgb(m.Color.G), Srgb(m.Color.B));
                _o.Rect(cx + 6, y0 + 6, 12, 12, c);
                _o.Frame(cx + 6, y0 + 6, 12, 12, new Rgba(0, 0, 0, 0.6f));
                string key = i == 9 ? "0" : (i + 1).ToString();
                float adv = _o.ShadowText(sel ? _f.UiBold : _f.Ui, $"{key} {m.Name}", cx + 22, y0 + 3, sel ? Text : Dim);
                int live = s.World.Group(m.Id).Count;
                if (live > 0) _o.ShadowText(_f.Mono, Compact(live), cx + 26 + adv, y0 + 5, sel ? Accent : new Rgba(0.55f, 0.6f, 0.68f));
            }
        }

        /// <summary>The help sheet (toggle with H).</summary>
        private void DrawHelp(in HudState s)
        {
            string[] lines =
            {
                "Mouse",
                "  Left drag            pour the selected material (throw it by moving)",
                "  Right drag           stir — push everything along with the cursor",
                "  Middle / Ctrl+Left   erase particles",
                "  Shift+Left / Right   paint / remove stone walls",
                "  Wheel  (+ / -)       brush size",
                "Keys",
                "  1-9, 0               Water Oil Honey Lava Sand Snow Jelly Rubber Clay Rock",
                "  F1 … F10             scenes          R  restart scene     C  clear particles",
                "  F                    faucet at cursor (Shift+F removes all faucets)",
                "  ← →                  tilt gravity    G  reset gravity     Space  pause",
                "  T                    transfers: fused SIMD ↔ pure NumSharp (np.bincount / np.take)",
                "  Q                    grid quality    V  vsync             F11  fullscreen",
                "  F12                  screenshot      H  this help         Esc  quit",
                "Physics",
                "  MLS-MPM: particles carry mass, velocity, deformation; a grid resolves momentum.",
                "  Every material law (fluid pressure + viscosity, Drucker-Prager sand, snow",
                "  plasticity, corotated elasticity, von Mises clay) runs as fused NumSharp kernels.",
            };
            float lh = 18, w = 640, h = lines.Length * lh + 24;
            float x = (s.Width - w) / 2, y = (s.Height - h) / 2;
            _o.Rect(x, y, w, h, new Rgba(0.015f, 0.02f, 0.03f, 0.88f));
            _o.Frame(x, y, w, h, new Rgba(1, 1, 1, 0.15f));
            for (int i = 0; i < lines.Length; i++)
            {
                bool head = !lines[i].StartsWith(" ", StringComparison.Ordinal);
                _o.ShadowText(head ? _f.UiBold : _f.Mono, lines[i], x + 16, y + 12 + i * lh, head ? Accent : Text);
            }
        }

        /// <summary>Horizontally centered text.</summary>
        private void Centered(FontFace face, string text, float y, Rgba c, int width)
        {
            float w = face.Measure(text);
            _o.Rect((width - w) / 2 - 12, y - 4, w + 24, face.LineHeight + 8, new Rgba(0, 0, 0, 0.45f));
            _o.ShadowText(face, text, (width - w) / 2, y, c);
        }

        /// <summary>1,234 → "1.2K".</summary>
        private static string Compact(int n) => n >= 1000 ? $"{n / 1000.0:F1}K" : n.ToString();

        /// <summary>Linear → sRGB for UI swatches (the scene is tone-mapped; the HUD is not).</summary>
        private static float Srgb(float linear) => MathF.Pow(Math.Clamp(linear, 0f, 1f), 1f / 2.2f);
    }
}
