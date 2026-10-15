using System.Text;
using UnityEngine;
using NumSharp.Examples.GravitySandbox.Physics;

namespace NumSharp.Examples.GravitySandbox
{
    /// <summary>
    /// Everything the HUD needs to draw for one frame, bundled so <see cref="GravitySandbox"/> hands the
    /// renderer a snapshot rather than the live world. Passing a value snapshot keeps the HUD a pure
    /// display function and avoids it reading half-updated state.
    /// </summary>
    public struct HudInfo
    {
        /// <summary>The active scenario's title.</summary>
        public string ScenarioName;
        /// <summary>The active scenario's one-line description.</summary>
        public string ScenarioDescription;
        /// <summary>The current conserved-quantity snapshot.</summary>
        public SystemDiagnostics Diagnostics;
        /// <summary>Total energy at scenario start, the baseline the live energy is compared against.</summary>
        public double InitialEnergy;
        /// <summary>Current body count (changes as bodies spawn or merge).</summary>
        public int BodyCount;
        /// <summary>Elapsed simulated time in scenario units.</summary>
        public double SimTime;
        /// <summary>Smoothed frames per second, for a performance readout.</summary>
        public float Fps;
        /// <summary>The integration scheme in use.</summary>
        public IntegrationMethod Method;
        /// <summary>The force model in use.</summary>
        public ForceModel Model;
        /// <summary>The current time-scale multiplier.</summary>
        public double TimeScale;
        /// <summary>Whether collision/merge is enabled.</summary>
        public bool Collisions;
        /// <summary>Whether the simulation is paused.</summary>
        public bool Paused;
    }

    /// <summary>
    /// Renders the on-screen instrument panel and controls legend with Unity's immediate-mode GUI. Being a
    /// plain class called from <see cref="GravitySandbox.OnGUI"/>, it needs no scene setup — which is the
    /// whole point of the sample: one script, no prefabs, no canvas. The panel foregrounds the conserved
    /// quantities because watching them hold constant is the evidence that the NumSharp-backed physics is
    /// faithful.
    /// </summary>
    public sealed class SandboxHud
    {
        private GUIStyle _panel;
        private GUIStyle _label;
        private GUIStyle _title;
        private readonly StringBuilder _sb = new StringBuilder(512);

        /// <summary>
        /// Lazily builds the GUI styles. GUIStyles can only be constructed inside the OnGUI call, so this is
        /// deferred out of the constructor and guarded by a null check.
        /// </summary>
        private void EnsureStyles()
        {
            if (_panel != null) return;

            // A translucent dark box so the readout stays legible over bright bodies and trails.
            var bg = new Texture2D(1, 1);
            bg.SetPixel(0, 0, new Color(0f, 0f, 0f, 0.55f));
            bg.Apply();

            _panel = new GUIStyle(GUI.skin.box) { normal = { background = bg } };
            _label = new GUIStyle(GUI.skin.label)
            {
                fontSize = 12,
                richText = true,
                alignment = TextAnchor.UpperLeft,
                normal = { textColor = Color.white },
            };
            _title = new GUIStyle(_label) { fontSize = 15, fontStyle = FontStyle.Bold };
        }

        /// <summary>
        /// Draws the whole HUD for one frame: a diagnostics panel on the left and a controls legend on the
        /// right. Call from <see cref="MonoBehaviour"/>.OnGUI only.
        /// </summary>
        /// <param name="info">The snapshot to display.</param>
        public void Draw(in HudInfo info)
        {
            EnsureStyles();
            DrawDiagnostics(in info);
            DrawControls();
        }

        /// <summary>Draws the left-hand panel: scenario, conserved quantities, energy drift, and status.</summary>
        /// <param name="info">The snapshot to display.</param>
        private void DrawDiagnostics(in HudInfo info)
        {
            var d = info.Diagnostics;
            // Relative energy drift from the scenario's starting energy — the headline "is it faithful?"
            // number. Guarded against a zero baseline (a system that happens to start at zero energy).
            double drift = info.InitialEnergy != 0 ? (d.Total - info.InitialEnergy) / System.Math.Abs(info.InitialEnergy) : 0.0;

            _sb.Clear();
            _sb.AppendLine($"<b>{info.ScenarioName}</b>");
            _sb.AppendLine($"<size=11>{info.ScenarioDescription}</size>");
            _sb.AppendLine();
            _sb.AppendLine($"Bodies:        {info.BodyCount}");
            _sb.AppendLine($"Sim time:      {info.SimTime:F2}");
            _sb.AppendLine($"FPS:           {info.Fps:F0}");
            _sb.AppendLine();
            _sb.AppendLine("<b>Conserved quantities (NumSharp)</b>");
            _sb.AppendLine($"Total energy:  {d.Total: 0.000e+00;-0.000e+00}");
            _sb.AppendLine($"  drift:       {drift: 0.00e+00;-0.00e+00}   (→ 0 if faithful)");
            _sb.AppendLine($"  kinetic:     {d.Kinetic: 0.000e+00}");
            _sb.AppendLine($"  potential:   {d.Potential: 0.000e+00}");
            _sb.AppendLine($"|momentum|:    {d.LinearMomentum.Length: 0.000e+00}");
            _sb.AppendLine($"|ang.mom.|:    {d.AngularMomentum.Length: 0.000e+00}");
            _sb.AppendLine($"total mass:    {d.TotalMass: 0.000e+00}");
            _sb.AppendLine();
            _sb.AppendLine("<b>Settings</b>");
            _sb.AppendLine($"Integrator:    {info.Method}");
            _sb.AppendLine($"Force model:   {info.Model}");
            _sb.AppendLine($"Time scale:    {info.TimeScale:0.##}x {(info.Paused ? "<color=#ff6>[PAUSED]</color>" : "")}");
            _sb.AppendLine($"Collisions:    {(info.Collisions ? "on" : "off")}");

            const float w = 340f, h = 420f;
            GUI.Box(new Rect(10, 10, w, h), GUIContent.none, _panel);
            GUI.Label(new Rect(22, 20, w - 24, h - 20), _sb.ToString(), _label);
        }

        /// <summary>Draws the right-hand controls legend.</summary>
        private void DrawControls()
        {
            _sb.Clear();
            _sb.AppendLine("<b>Controls</b>");
            _sb.AppendLine("1–6          load scenario");
            _sb.AppendLine("Space        pause / resume");
            _sb.AppendLine("R            reset scenario");
            _sb.AppendLine("[ / ]        slower / faster time");
            _sb.AppendLine("I            cycle integrator");
            _sb.AppendLine("G            toggle relativity");
            _sb.AppendLine("C            toggle collisions");
            _sb.AppendLine();
            _sb.AppendLine("<b>Camera</b>");
            _sb.AppendLine("Right-drag   orbit");
            _sb.AppendLine("Middle-drag  pan");
            _sb.AppendLine("Scroll       zoom");
            _sb.AppendLine();
            _sb.AppendLine("<b>Create a body</b>");
            _sb.AppendLine("Left-drag    slingshot:");
            _sb.AppendLine("             click at start point,");
            _sb.AppendLine("             drag to set velocity,");
            _sb.AppendLine("             release to launch.");

            const float w = 260f, h = 330f;
            float x = Screen.width - w - 10f;
            GUI.Box(new Rect(x, 10, w, h), GUIContent.none, _panel);
            GUI.Label(new Rect(x + 12, 20, w - 24, h - 20), _sb.ToString(), _label);
        }
    }
}
