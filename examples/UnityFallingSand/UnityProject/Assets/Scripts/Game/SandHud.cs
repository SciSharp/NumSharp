using System.Text;
using UnityEngine;
using NumSharp.Examples.FallingSand.Simulation;

namespace NumSharp.Examples.FallingSand
{
    /// <summary>What the HUD wants the game to do this frame, returned from <see cref="SandHud.Draw"/>.</summary>
    public struct SandHudResult
    {
        /// <summary>The material the palette says should be selected (unchanged if no button was clicked).</summary>
        public int SelectedMaterial;
        /// <summary>True if the player clicked "Clear" (clear loose material, keep walls).</summary>
        public bool ClearRequested;
        /// <summary>True if the player clicked "Clear All" (wipe everything including walls).</summary>
        public bool ClearAllRequested;
    }

    /// <summary>
    /// The immediate-mode overlay: a clickable material palette on the left and a stats/controls panel on
    /// the right. As a plain class called from <see cref="FallingSandGame.OnGUI"/> it needs no canvas or
    /// prefab. It also DEFINES the left-hand no-paint strip (<see cref="PaletteWidth"/>) the game uses to
    /// keep palette clicks from also painting into the world.
    /// </summary>
    public sealed class SandHud
    {
        /// <summary>Width in pixels of the left palette strip; the game treats this region as UI, not canvas.</summary>
        public const float PaletteWidth = 150f;

        // Palette entries (material id + label), in display order.
        private static readonly (int id, string label)[] Palette =
        {
            (Cell.Sand, "Sand"), (Cell.Water, "Water"), (Cell.Oil, "Oil"),
            (Cell.Smoke, "Smoke"), (Cell.Wall, "Wall"), (Cell.Empty, "Erase"),
        };

        private GUIStyle _panel, _label, _title, _btn, _btnSel;
        private readonly StringBuilder _sb = new StringBuilder(512);

        /// <summary>Builds the GUI styles once (must happen inside OnGUI).</summary>
        private void EnsureStyles()
        {
            if (_panel != null) return;
            var bg = new Texture2D(1, 1);
            bg.SetPixel(0, 0, new Color(0f, 0f, 0f, 0.6f));
            bg.Apply();
            _panel = new GUIStyle(GUI.skin.box) { normal = { background = bg } };
            _label = new GUIStyle(GUI.skin.label) { fontSize = 12, richText = true, normal = { textColor = Color.white } };
            _title = new GUIStyle(_label) { fontSize = 14, fontStyle = FontStyle.Bold };
            _btn = new GUIStyle(GUI.skin.button) { fontSize = 12 };
            _btnSel = new GUIStyle(_btn) { fontStyle = FontStyle.Bold };   // selected entry drawn bold + marked
        }

        /// <summary>
        /// Draws the palette and the stats/controls panel, returning any requested action. Call only from OnGUI.
        /// </summary>
        /// <param name="world">The world (read for live material counts).</param>
        /// <param name="selected">The currently selected material id (to highlight).</param>
        /// <param name="brushRadius">Current brush radius (shown).</param>
        /// <param name="subSteps">Current physics substeps per frame (shown).</param>
        /// <param name="paused">Whether the sim is paused (shown).</param>
        /// <param name="fps">Smoothed FPS (shown).</param>
        /// <returns>The (possibly changed) selection and any clear request.</returns>
        public SandHudResult Draw(FallingSandWorld world, int selected, int brushRadius, int subSteps, bool paused, float fps)
        {
            EnsureStyles();
            var result = new SandHudResult { SelectedMaterial = selected };

            // ---- left: material palette ----
            float y = 10f;
            GUI.Box(new Rect(6, 6, PaletteWidth - 12, Palette.Length * 34 + 84), GUIContent.none, _panel);
            GUI.Label(new Rect(16, y, PaletteWidth - 24, 20), "<b>Material</b>", _title); y += 24;
            foreach (var (id, label) in Palette)
            {
                bool isSel = id == selected;
                string text = isSel ? "▶ " + label : label;
                if (GUI.Button(new Rect(16, y, PaletteWidth - 32, 28), text, isSel ? _btnSel : _btn))
                    result.SelectedMaterial = id;
                y += 32;
            }
            if (GUI.Button(new Rect(16, y, (PaletteWidth - 40) / 2, 26), "Clear")) result.ClearRequested = true;
            if (GUI.Button(new Rect(16 + (PaletteWidth - 40) / 2 + 6, y, (PaletteWidth - 40) / 2, 26), "Wipe")) result.ClearAllRequested = true;

            // ---- right: stats + controls ----
            _sb.Clear();
            _sb.AppendLine("<b>Falling Sand — NumSharp physics</b>");
            _sb.AppendLine($"grid {world.Width}×{world.Height}   fps {fps:F0}   {(paused ? "<color=#ff6>[PAUSED]</color>" : "")}");
            _sb.AppendLine($"brush r={brushRadius}   substeps={subSteps}   faucets={world.EmitterCount}");
            _sb.AppendLine();
            _sb.AppendLine("<b>Counts</b>");
            _sb.AppendLine($"sand {world.Count(Cell.Sand)}   water {world.Count(Cell.Water)}");
            _sb.AppendLine($"oil {world.Count(Cell.Oil)}   smoke {world.Count(Cell.Smoke)}   wall {world.Count(Cell.Wall)}");
            _sb.AppendLine();
            _sb.AppendLine("<b>Controls</b>");
            _sb.AppendLine("1-6 pick material   ·   left-drag paint / pour");
            _sb.AppendLine("[ ] brush size   ·   , . substeps   ·   Space pause");
            _sb.AppendLine("E place faucet   ·   F clear faucets   ·   C clear   ·   Shift+C wipe");

            const float w = 560f, h = 150f;
            float x = Screen.width - w - 10f;
            GUI.Box(new Rect(x, 6, w, h), GUIContent.none, _panel);
            GUI.Label(new Rect(x + 12, 12, w - 24, h - 12), _sb.ToString(), _label);

            return result;
        }
    }
}
