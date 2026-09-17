using System.Text;
using UnityEngine;
using NumSharp.Examples.FallingSand.Simulation;

namespace NumSharp.Examples.FallingSand
{
    /// <summary>What the HUD wants the game to do this frame, returned from <see cref="SandHud.Draw"/>.</summary>
    /// <remarks>
    /// The HUD is stateless: the game owns the brush/selection state and passes it in; the HUD echoes it back
    /// here with any click applied, so the game stays the single source of truth. A value that a control did
    /// NOT touch comes back unchanged (the same value the game passed in), which is why the game can apply
    /// every field unconditionally.
    /// </remarks>
    public struct SandHudResult
    {
        /// <summary>The material the palette says should be selected (unchanged if no button was clicked).</summary>
        public int SelectedMaterial;
        /// <summary>The brush radius after any +/- click (unchanged otherwise).</summary>
        public int BrushRadius;
        /// <summary>The brush shape after any shape-toggle click (unchanged otherwise).</summary>
        public BrushShape BrushShape;
        /// <summary>True if the player clicked "Clear" (clear loose material, keep walls).</summary>
        public bool ClearRequested;
        /// <summary>True if the player clicked "Wipe" (wipe everything including walls).</summary>
        public bool ClearAllRequested;
        /// <summary>The template index to load this frame, or -1 if no template icon was clicked.</summary>
        public int TemplateRequested;
    }

    /// <summary>
    /// The immediate-mode overlay: a clickable material palette and brush controls on the left, a strip of
    /// template icons along the bottom, and a stats/controls panel on the right. As a plain class called from
    /// <see cref="FallingSandGame.OnGUI"/> it needs no canvas or prefab. It also DEFINES the two no-paint
    /// regions (<see cref="PaletteWidth"/>, <see cref="TemplateStripHeight"/>) the game uses to keep HUD
    /// clicks from also painting into the world.
    /// </summary>
    /// <remarks>
    /// The template icons are not image assets — there are none in this sample. Each is generated once by
    /// building that template into a tiny world, stepping it a few frames so it reaches a recognizable shape,
    /// and colouring the result through the same <see cref="Cell.Color"/> table the game renders with. So an
    /// icon is a genuine miniature of the scene it loads.
    /// </remarks>
    public sealed class SandHud
    {
        /// <summary>Width in pixels of the left palette strip; the game treats this region as UI, not canvas.</summary>
        public const float PaletteWidth = 168f;

        /// <summary>Height in pixels of the bottom template strip; the game treats this region as UI, not canvas.</summary>
        public const float TemplateStripHeight = 82f;

        // Palette entries (material id + label), in display order — now including fire and lava.
        private static readonly (int id, string label)[] Palette =
        {
            (Cell.Sand, "Sand"), (Cell.Water, "Water"), (Cell.Oil, "Oil"), (Cell.Smoke, "Smoke"),
            (Cell.Fire, "Fire"), (Cell.Lava, "Lava"), (Cell.Wall, "Wall"), (Cell.Empty, "Erase"),
        };

        private GUIStyle _panel, _label, _title, _small, _btn, _btnSel, _iconHit;
        private readonly StringBuilder _sb = new StringBuilder(512);
        private Texture2D[] _icons;   // one generated thumbnail per template (built once)

        /// <summary>Builds the GUI styles and the template thumbnails once (must happen inside OnGUI, where the GUI skin and texture APIs are valid).</summary>
        private void EnsureStyles()
        {
            if (_panel != null) return;
            var bg = new Texture2D(1, 1);
            bg.SetPixel(0, 0, new Color(0f, 0f, 0f, 0.6f));
            bg.Apply();
            _panel = new GUIStyle(GUI.skin.box) { normal = { background = bg } };
            _label = new GUIStyle(GUI.skin.label) { fontSize = 12, richText = true, normal = { textColor = Color.white } };
            _title = new GUIStyle(_label) { fontSize = 14, fontStyle = FontStyle.Bold };
            _small = new GUIStyle(_label) { fontSize = 10, alignment = TextAnchor.UpperCenter };
            _btn = new GUIStyle(GUI.skin.button) { fontSize = 12 };
            _btnSel = new GUIStyle(_btn) { fontStyle = FontStyle.Bold };   // selected entry drawn bold + marked
            _iconHit = new GUIStyle();                                     // invisible click target laid over an icon
            _icons = BuildTemplateIcons();
        }

        /// <summary>
        /// Generates the template thumbnails: for each template, builds it into a small world, steps it so the
        /// scene fills in, and colours the snapshot into a point-filtered texture. Wrapped defensively so a
        /// problem building any one icon degrades to a flat placeholder rather than breaking the whole HUD.
        /// </summary>
        /// <returns>One <see cref="Texture2D"/> per template, index-aligned with <see cref="SandScenes.Names"/>.</returns>
        private static Texture2D[] BuildTemplateIcons()
        {
            const int iw = 64, ih = 40;
            var lut = BuildLut();
            var icons = new Texture2D[FallingSandWorld.TemplateCount];
            for (int t = 0; t < icons.Length; t++)
            {
                var tex = new Texture2D(iw, ih, TextureFormat.RGBA32, false) { filterMode = FilterMode.Point, wrapMode = TextureWrapMode.Clamp };
                var px = new Color32[iw * ih];
                try
                {
                    var w = new FallingSandWorld(ih, iw, seed: 1, withBoundary: true);
                    w.LoadTemplate(t);
                    w.Step(36);   // let material fall/flow into a recognizable silhouette
                    var snap = w.Grid.Snapshot();
                    for (int r = 0; r < ih; r++)
                        for (int c = 0; c < iw; c++)
                            px[(ih - 1 - r) * iw + c] = lut[snap[r * iw + c]];   // vertical flip, grid top → texture top
                }
                catch
                {
                    for (int i = 0; i < px.Length; i++) px[i] = new Color32(30, 30, 40, 255);   // safe placeholder
                }
                tex.SetPixels32(px);
                tex.Apply(false);
                icons[t] = tex;
            }
            return icons;
        }

        /// <summary>Builds the material→colour lookup (0..1 floats → 0..255 bytes) from <see cref="Cell.Color"/>.</summary>
        /// <returns>A colour per material id.</returns>
        private static Color32[] BuildLut()
        {
            var lut = new Color32[Cell.Count];
            for (int id = 0; id < Cell.Count; id++)
            {
                var c = Cell.Color[id];
                lut[id] = new Color32((byte)(c.r * 255), (byte)(c.g * 255), (byte)(c.b * 255), 255);
            }
            return lut;
        }

        /// <summary>
        /// Draws the palette, brush controls, template strip and stats panel, returning any requested action.
        /// Call only from OnGUI.
        /// </summary>
        /// <param name="world">The world (read for live material counts).</param>
        /// <param name="selected">The currently selected material id (to highlight).</param>
        /// <param name="brushRadius">Current brush radius (shown; adjustable via the +/- buttons).</param>
        /// <param name="brushShape">Current brush shape (shown; toggled by the shape button).</param>
        /// <param name="subSteps">Current physics substeps per frame (shown).</param>
        /// <param name="paused">Whether the sim is paused (shown).</param>
        /// <param name="fps">Smoothed FPS (shown).</param>
        /// <returns>The (possibly changed) selection, brush radius/shape, any clear request, and any template pick.</returns>
        public SandHudResult Draw(FallingSandWorld world, int selected, int brushRadius, BrushShape brushShape, int subSteps, bool paused, float fps)
        {
            EnsureStyles();
            var result = new SandHudResult
            {
                SelectedMaterial = selected,
                BrushRadius = brushRadius,
                BrushShape = brushShape,
                TemplateRequested = -1,
            };

            // ---- left: material palette + brush controls ----
            float panelH = Palette.Length * 32 + 150;
            GUI.Box(new Rect(6, 6, PaletteWidth - 12, panelH), GUIContent.none, _panel);
            float y = 12f;
            GUI.Label(new Rect(16, y, PaletteWidth - 24, 20), "<b>Material</b>", _title); y += 24;
            foreach (var (id, label) in Palette)
            {
                bool isSel = id == selected;
                string text = isSel ? "▶ " + label : label;
                if (GUI.Button(new Rect(16, y, PaletteWidth - 32, 28), text, isSel ? _btnSel : _btn))
                    result.SelectedMaterial = id;
                y += 32;
            }

            // Brush size: − [value] + , then a shape toggle.
            y += 6;
            GUI.Label(new Rect(16, y, PaletteWidth - 24, 20), "<b>Brush</b>", _title); y += 22;
            if (GUI.Button(new Rect(16, y, 30, 26), "−")) result.BrushRadius = Mathf.Max(0, result.BrushRadius - 1);
            GUI.Label(new Rect(50, y + 4, 60, 20), $"size {result.BrushRadius}", _label);
            if (GUI.Button(new Rect(PaletteWidth - 46, y, 30, 26), "+")) result.BrushRadius = Mathf.Min(40, result.BrushRadius + 1);
            y += 30;
            if (GUI.Button(new Rect(16, y, PaletteWidth - 32, 26), $"shape: {result.BrushShape}"))
                result.BrushShape = result.BrushShape == BrushShape.Disk ? BrushShape.Square : BrushShape.Disk;
            y += 32;

            // Clear / Wipe.
            if (GUI.Button(new Rect(16, y, (PaletteWidth - 40) / 2, 26), "Clear")) result.ClearRequested = true;
            if (GUI.Button(new Rect(16 + (PaletteWidth - 40) / 2 + 6, y, (PaletteWidth - 40) / 2, 26), "Wipe")) result.ClearAllRequested = true;

            // ---- bottom: template icon strip ----
            DrawTemplateStrip(ref result);

            // ---- right: stats + controls ----
            _sb.Clear();
            _sb.AppendLine("<b>Falling Sand — NumSharp physics</b>");
            _sb.AppendLine($"grid {world.Width}×{world.Height}   fps {fps:F0}   {(paused ? "<color=#ff6>[PAUSED]</color>" : "")}");
            _sb.AppendLine($"brush {result.BrushShape} r={result.BrushRadius}   substeps={subSteps}   faucets={world.EmitterCount}");
            _sb.AppendLine();
            _sb.AppendLine("<b>Counts</b>");
            _sb.AppendLine($"sand {world.Count(Cell.Sand)}   water {world.Count(Cell.Water)}   oil {world.Count(Cell.Oil)}");
            _sb.AppendLine($"smoke {world.Count(Cell.Smoke)}   fire {world.Count(Cell.Fire)}   lava {world.Count(Cell.Lava)}   wall {world.Count(Cell.Wall)}");
            _sb.AppendLine();
            _sb.AppendLine("<b>Controls</b>");
            _sb.AppendLine("1-8 pick material   ·   left-drag paint / pour   ·   B brush shape");
            _sb.AppendLine("[ ] brush size   ·   , . substeps   ·   Space pause");
            _sb.AppendLine("E place faucet · F clear faucets · C clear · Shift+C wipe · F1-F6 templates");

            const float w = 600f, h = 150f;
            float x = Screen.width - w - 10f;
            GUI.Box(new Rect(x, 6, w, h), GUIContent.none, _panel);
            GUI.Label(new Rect(x + 12, 12, w - 24, h - 12), _sb.ToString(), _label);

            return result;
        }

        /// <summary>
        /// Draws the row of clickable template thumbnails centred along the bottom. Each icon is the texture
        /// with an invisible button laid over it (so the miniature stays visible while still catching clicks);
        /// a click sets <see cref="SandHudResult.TemplateRequested"/> to that template's index.
        /// </summary>
        /// <param name="result">The result being assembled this frame; its template field is set on a click.</param>
        private void DrawTemplateStrip(ref SandHudResult result)
        {
            int nt = FallingSandWorld.TemplateCount;
            const float iw = 72f, ih = 46f, pad = 10f, labelH = 14f;
            float totalW = nt * iw + (nt - 1) * pad;
            float x0 = (Screen.width - totalW) * 0.5f;
            float yTop = Screen.height - TemplateStripHeight + 6f;

            GUI.Box(new Rect(x0 - 12, yTop - 6, totalW + 24, TemplateStripHeight - 4), GUIContent.none, _panel);
            var names = FallingSandWorld.TemplateNames;
            for (int t = 0; t < nt; t++)
            {
                var iconRect = new Rect(x0 + t * (iw + pad), yTop, iw, ih);
                if (_icons != null && t < _icons.Length && _icons[t] != null)
                    GUI.DrawTexture(iconRect, _icons[t]);
                GUI.Label(new Rect(iconRect.x, iconRect.yMax, iw, labelH), names[t], _small);
                // Invisible hit target over the icon — clickable without a button chrome hiding the thumbnail.
                if (GUI.Button(iconRect, GUIContent.none, _iconHit)) result.TemplateRequested = t;
            }
        }
    }
}
