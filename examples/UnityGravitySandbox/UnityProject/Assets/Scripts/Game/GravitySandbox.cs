using System;
using System.Collections.Generic;
using UnityEngine;
using NumSharp.Examples.GravitySandbox.Physics;

namespace NumSharp.Examples.GravitySandbox
{
    /// <summary>
    /// The single MonoBehaviour that runs the whole sample. Drop it on ONE empty GameObject and press Play:
    /// it builds its own camera, lights and bodies from code (no scene, prefab, or material asset needed),
    /// steps the NumSharp-backed <see cref="NBodyWorld"/> every frame, mirrors the physics into
    /// <see cref="BodyView"/>s, handles all input, and draws the <see cref="SandboxHud"/>. Keeping the whole
    /// game self-bootstrapping is deliberate — it makes the example reproducible from source alone, which
    /// matters when the point is to show NumSharp doing the physics, not to ship a Unity project layout.
    /// </summary>
    /// <remarks>
    /// <b>The frame loop</b> is: read input → advance the simulation by <c>timeScale × realΔt</c> of
    /// simulated time in fixed substeps (bounded so a slow frame can't cascade) → if the body set changed,
    /// rebuild the views → move every view to its body → cache a HUD snapshot. All the physics — pairwise
    /// gravity, integration, energy/momentum diagnostics — happens inside NumSharp array math; this class
    /// is purely the bridge to Unity.
    /// </remarks>
    [DisallowMultipleComponent]
    public sealed class GravitySandbox : MonoBehaviour
    {
        [Header("Rendering")]
        [Tooltip("Multiplier applied to each body's physical radius when drawing it.")]
        public float VisualScale = 1.0f;
        [Tooltip("Floor on rendered radius so distant/tiny bodies stay visible (cosmetic only; does not affect physics).")]
        public float MinRenderRadius = 0.12f;

        [Header("Simulation")]
        [Tooltip("Hard cap on physics substeps per frame, so a slow frame cannot spiral into an ever-growing backlog.")]
        public int MaxSubstepsPerFrame = 200;

        private Scenario[] _scenarios;
        private int _scenarioIndex;
        private NBodyWorld _world;

        private readonly List<BodyView> _views = new List<BodyView>();
        private int _viewVersion = -1;         // the world.Version the current views were built for
        private Transform _bodiesRoot;

        private Camera _cam;
        private CameraRig _rig;
        private SandboxHud _hud;

        private double _initialEnergy;          // energy at scenario start, the drift baseline
        private bool _paused;
        private float _fps = 60f;

        // Left-drag "slingshot" spawn state.
        private LineRenderer _sling;
        private bool _dragging;
        private Vector3 _dragStartWorld;
        private Vector2 _dragStartScreen;
        private readonly Plane _simPlane = new Plane(Vector3.forward, Vector3.zero);   // the z=0 orbital plane
        private static readonly Color[] SpawnPalette =
        {
            new Color(1f, 0.55f, 0.35f), new Color(0.45f, 0.85f, 1f), new Color(0.6f, 1f, 0.55f),
            new Color(1f, 0.85f, 0.4f), new Color(0.9f, 0.55f, 1f), new Color(1f, 0.45f, 0.6f),
        };
        private int _spawnColor;

        private HudInfo _hudInfo;   // snapshot rebuilt each Update, drawn each OnGUI

        /// <summary>
        /// Builds the entire scene from code: camera + rig, lighting, the slingshot preview line, the
        /// scenario set, and the first scenario's bodies. Runs once when the GameObject wakes.
        /// </summary>
        private void Start()
        {
            BootstrapCameraAndLighting();
            BootstrapSlingshotPreview();
            _hud = new SandboxHud();
            _bodiesRoot = new GameObject("Bodies").transform;

            _scenarios = ScenarioLibrary.All();
            _world = new NBodyWorld(_scenarios[0]);
            LoadScenario(0);
        }

        /// <summary>Creates the camera with an orbit rig and sets up ambient + key lighting so every scenario is visible.</summary>
        private void BootstrapCameraAndLighting()
        {
            // Reuse the scene's existing camera (a fresh Unity scene ships a "Main Camera") rather than
            // adding a second one, which would double-render and stack AudioListeners. Only create one if
            // the scene genuinely has none.
            _cam = Camera.main != null ? Camera.main : FindFirstObjectByType<Camera>();
            if (_cam == null)
                _cam = new GameObject("Sandbox Camera").AddComponent<Camera>();

            _cam.clearFlags = CameraClearFlags.SolidColor;
            _cam.backgroundColor = new Color(0.02f, 0.02f, 0.05f);   // near-black space
            _cam.nearClipPlane = 0.01f;
            _cam.farClipPlane = 5000f;
            if (_cam.GetComponent<AudioListener>() == null)
                _cam.gameObject.AddComponent<AudioListener>();       // silences the "no audio listener" warning
            _rig = _cam.GetComponent<CameraRig>();
            if (_rig == null) _rig = _cam.gameObject.AddComponent<CameraRig>();

            // Flat ambient guarantees non-star bodies (e.g. the figure-eight) are lit even with no star in
            // the scene; a soft directional key light adds shape to the spheres.
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.35f, 0.35f, 0.42f);
            var keyGo = new GameObject("Key Light");
            var key = keyGo.AddComponent<Light>();
            key.type = LightType.Directional;
            key.intensity = 0.6f;
            keyGo.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        }

        /// <summary>Creates the always-present (usually hidden) line that previews a slingshot launch vector.</summary>
        private void BootstrapSlingshotPreview()
        {
            var go = new GameObject("Slingshot Preview");
            _sling = go.AddComponent<LineRenderer>();
            _sling.useWorldSpace = true;
            _sling.positionCount = 2;
            _sling.widthMultiplier = MinRenderRadius * 0.4f;
            _sling.sharedMaterial = SandboxRenderKit.CreateTrailMaterial(Color.white);
            _sling.enabled = false;
        }

        /// <summary>
        /// Switches to scenario <paramref name="index"/>: reloads the world from that recipe, re-frames the
        /// camera, captures the fresh energy baseline, and forces a view rebuild. Wraps the index so the
        /// number keys are forgiving.
        /// </summary>
        /// <param name="index">The scenario index; wrapped into range.</param>
        private void LoadScenario(int index)
        {
            _scenarioIndex = ((index % _scenarios.Length) + _scenarios.Length) % _scenarios.Length;
            _world.LoadScenario(_scenarios[_scenarioIndex]);
            _initialEnergy = _world.Diagnostics().Total;
            _rig.Frame((float)_scenarios[_scenarioIndex].CameraDistance);
            _paused = false;
            _viewVersion = -1;   // invalidate views so SyncViews rebuilds against the new body set
            SyncViews();
        }

        /// <summary>Per-frame driver: input, physics advance, view sync, and HUD snapshot.</summary>
        private void Update()
        {
            HandleKeyboard();
            HandleSlingshot();

            // Advance simulated time proportional to real time; fixed substeps inside keep the integration
            // stable and frame-rate independent, and the cap protects the frame budget.
            if (!_paused)
            {
                double simSeconds = _world.TimeScale * Time.deltaTime;
                _world.Advance(simSeconds, MaxSubstepsPerFrame);
            }

            SyncViews();

            // Smooth the FPS readout so it doesn't flicker frame to frame.
            _fps = Mathf.Lerp(_fps, 1f / Mathf.Max(Time.unscaledDeltaTime, 1e-5f), 0.1f);
            CacheHudInfo();
        }

        /// <summary>Reads the keyboard shortcuts and applies them (scenario switch, pause, reset, tuning).</summary>
        private void HandleKeyboard()
        {
            // Number keys 1..N load scenarios.
            for (int k = 0; k < _scenarios.Length && k < 9; k++)
                if (Input.GetKeyDown(KeyCode.Alpha1 + k))
                    LoadScenario(k);

            if (Input.GetKeyDown(KeyCode.Space)) _paused = !_paused;
            if (Input.GetKeyDown(KeyCode.R)) LoadScenario(_scenarioIndex);   // reset current scenario

            // Time scale down/up. Clamped so it can't hit zero or run away.
            if (Input.GetKeyDown(KeyCode.LeftBracket)) _world.TimeScale = Math.Max(_world.TimeScale * 0.7, 1e-3);
            if (Input.GetKeyDown(KeyCode.RightBracket)) _world.TimeScale = Math.Min(_world.TimeScale * 1.4, 1e6);

            if (Input.GetKeyDown(KeyCode.I)) CycleIntegrator();
            if (Input.GetKeyDown(KeyCode.G)) ToggleRelativity();
            if (Input.GetKeyDown(KeyCode.C)) _world.EnableCollisions = !_world.EnableCollisions;
        }

        /// <summary>Advances the integration scheme to the next of the three, invalidating the cached acceleration.</summary>
        private void CycleIntegrator()
        {
            _world.Integrator.Method = _world.Integrator.Method switch
            {
                IntegrationMethod.SymplecticEuler => IntegrationMethod.Leapfrog,
                IntegrationMethod.Leapfrog => IntegrationMethod.Rk4,
                _ => IntegrationMethod.SymplecticEuler,
            };
            _world.Integrator.Reset();   // the new scheme must not reuse the old one's cached state
        }

        /// <summary>
        /// Flips between Newtonian and relativistic force models. When switching ON with no finite light
        /// speed set (most scenarios leave it infinite), it picks a light speed a bit above the fastest
        /// body so the precession is actually visible rather than the real-world imperceptible amount.
        /// </summary>
        private void ToggleRelativity()
        {
            if (_world.Solver.Model == ForceModel.Newtonian)
            {
                if (!double.IsFinite(_world.Solver.SpeedOfLight))
                    _world.Solver.SpeedOfLight = Math.Max(MaxSpeed() * 15.0, 1e-6);  // v/c ≈ 0.07 → visible drift
                _world.Solver.Model = ForceModel.RelativisticPrecession;
            }
            else
            {
                _world.Solver.Model = ForceModel.Newtonian;
            }
            _world.Integrator.Reset();   // force law changed → cached acceleration is stale
        }

        /// <summary>
        /// The left-drag slingshot: press to place a start point on the orbital plane, drag to aim and set
        /// speed, release to spawn a new body launched in the drag direction. The launch speed is scaled by
        /// the on-SCREEN drag length against a characteristic orbital speed of the current system, so it
        /// feels the same regardless of the scenario's unit system.
        /// </summary>
        private void HandleSlingshot()
        {
            if (Input.GetMouseButtonDown(0) && !IsPointerOverHud(Input.mousePosition))
            {
                if (TryPlanePoint(Input.mousePosition, out _dragStartWorld))
                {
                    _dragging = true;
                    _dragStartScreen = Input.mousePosition;
                    _sling.enabled = true;
                    _sling.SetPosition(0, _dragStartWorld);
                    _sling.SetPosition(1, _dragStartWorld);
                }
            }
            else if (_dragging && Input.GetMouseButton(0))
            {
                if (TryPlanePoint(Input.mousePosition, out var cur))
                    _sling.SetPosition(1, cur);   // live-preview the launch vector
            }
            else if (_dragging && Input.GetMouseButtonUp(0))
            {
                _dragging = false;
                _sling.enabled = false;
                if (TryPlanePoint(Input.mousePosition, out var end))
                    SpawnFromDrag(_dragStartWorld, end, (Vector2)Input.mousePosition - _dragStartScreen);
            }
        }

        /// <summary>
        /// Spawns a body at the drag's start with a velocity in the drag DIRECTION whose magnitude scales
        /// with the on-screen drag length relative to a characteristic orbital speed. Mass and size are
        /// chosen relative to the current system so the spawned body reads as a small companion.
        /// </summary>
        /// <param name="startWorld">World-space launch point (on the orbital plane).</param>
        /// <param name="endWorld">World-space drag end (gives the launch direction).</param>
        /// <param name="screenDrag">Screen-space drag vector (gives the launch speed).</param>
        private void SpawnFromDrag(Vector3 startWorld, Vector3 endWorld, Vector2 screenDrag)
        {
            Vector3 dir = endWorld - startWorld;
            if (dir.sqrMagnitude < 1e-12f) dir = Vector3.right;   // a click with no drag → arbitrary but valid
            dir.Normalize();

            double maxMass = MaxMass();
            double camDist = _scenarios[_scenarioIndex].CameraDistance;
            // Characteristic circular speed at a quarter of the view distance — a natural "one orbit" scale.
            double vRef = Math.Sqrt(_world.Solver.G * Math.Max(maxMass, 1e-30) / Math.Max(0.25 * camDist, 1e-6));
            // A drag of ~30% of the screen height maps to that reference speed.
            double speed = (screenDrag.magnitude / (0.3f * Screen.height)) * vRef;

            var vel = new Vector3d(dir.x * speed, dir.y * speed, 0);   // launch stays in the orbital plane
            var pos = new Vector3d(startWorld.x, startWorld.y, 0);
            double mass = Math.Max(maxMass * 1e-3, 1e-30);             // a light companion relative to the primary
            double radius = 0.01 * camDist;                           // display/collision radius scaled to the scene

            Color c = SpawnPalette[_spawnColor++ % SpawnPalette.Length];
            var desc = new BodyDescriptor($"Spawn{_world.State.Count}", c.r, c.g, c.b, radius);
            _world.AddBody(pos, vel, mass, desc);   // bumps Version → SyncViews rebuilds next frame
        }

        /// <summary>Rebuilds the body views if the world's topology changed, then moves every view to its body.</summary>
        private void SyncViews()
        {
            if (_viewVersion != _world.Version)
            {
                // Topology changed (load/reset/spawn/merge): discard and rebuild the whole view set so
                // indices line up with the new body arrays. Cheap for the body counts used here.
                foreach (var v in _views) v.Dispose();
                _views.Clear();

                var descs = _world.Descriptors;
                int trail = _world.CurrentScenario.TrailLength;
                for (int i = 0; i < _world.State.Count; i++)
                    _views.Add(new BodyView(i, descs[i], _bodiesRoot, trail, VisualScale, MinRenderRadius));

                _viewVersion = _world.Version;
            }

            // Move each sphere/trail to its body's current position.
            for (int i = 0; i < _views.Count; i++)
            {
                var p = _world.State.PositionOf(i);
                _views[i].UpdateVisual(new Vector3((float)p.X, (float)p.Y, (float)p.Z));
            }
        }

        /// <summary>Captures a HUD snapshot from the current world state so OnGUI can draw without touching live data.</summary>
        private void CacheHudInfo()
        {
            var s = _scenarios[_scenarioIndex];
            _hudInfo = new HudInfo
            {
                ScenarioName = s.Name,
                ScenarioDescription = s.Description,
                Diagnostics = _world.Diagnostics(),
                InitialEnergy = _initialEnergy,
                BodyCount = _world.State.Count,
                SimTime = _world.Time,
                Fps = _fps,
                Method = _world.Integrator.Method,
                Model = _world.Solver.Model,
                TimeScale = _world.TimeScale,
                Collisions = _world.EnableCollisions,
                Paused = _paused,
            };
        }

        /// <summary>Draws the HUD. Unity may call OnGUI several times per frame, so it only renders the cached snapshot.</summary>
        private void OnGUI()
        {
            if (_hud != null) _hud.Draw(in _hudInfo);
        }

        /// <summary>Destroys every runtime-created view (and its materials/GameObjects) when the sandbox is torn down.</summary>
        private void OnDestroy()
        {
            foreach (var v in _views) v.Dispose();
            _views.Clear();
        }

        /// <summary>Ray-casts a screen point onto the z=0 orbital plane.</summary>
        /// <param name="screen">The screen-space point (e.g. mouse position).</param>
        /// <param name="world">Receives the world-space intersection with the plane.</param>
        /// <returns>True if the ray hit the plane in front of the camera; false otherwise.</returns>
        private bool TryPlanePoint(Vector3 screen, out Vector3 world)
        {
            Ray ray = _cam.ScreenPointToRay(screen);
            if (_simPlane.Raycast(ray, out float enter))
            {
                world = ray.GetPoint(enter);
                return true;
            }
            world = Vector3.zero;
            return false;
        }

        /// <summary>
        /// Approximates whether a screen point is over one of the HUD panels, so clicking the UI does not
        /// also spawn a body. Uses the known panel rectangles (converted from GUI's top-left origin to the
        /// input system's bottom-left origin).
        /// </summary>
        /// <param name="mouse">Mouse position in input coordinates (origin bottom-left).</param>
        /// <returns>True if the point falls within a HUD panel.</returns>
        private bool IsPointerOverHud(Vector3 mouse)
        {
            float guiY = Screen.height - mouse.y;   // input Y is bottom-up; GUI rects are top-down
            var left = new Rect(10, 10, 340, 420);
            var right = new Rect(Screen.width - 270, 10, 260, 330);
            return left.Contains(new Vector2(mouse.x, guiY)) || right.Contains(new Vector2(mouse.x, guiY));
        }

        /// <summary>The largest body mass currently in the system (used to scale spawns and the light-speed default).</summary>
        /// <returns>The maximum mass, or 0 for an empty system.</returns>
        private double MaxMass()
        {
            double m = 0;
            for (int i = 0; i < _world.State.Count; i++) m = Math.Max(m, _world.State.MassOf(i));
            return m;
        }

        /// <summary>The largest body speed currently in the system (used to choose a visible relativistic light speed).</summary>
        /// <returns>The maximum speed, or 0 for an empty system.</returns>
        private double MaxSpeed()
        {
            double s = 0;
            for (int i = 0; i < _world.State.Count; i++) s = Math.Max(s, _world.State.VelocityOf(i).Length);
            return s;
        }
    }
}
