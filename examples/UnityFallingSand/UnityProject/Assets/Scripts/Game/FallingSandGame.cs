using UnityEngine;
using NumSharp.Examples.FallingSand.Simulation;

namespace NumSharp.Examples.FallingSand
{
    /// <summary>
    /// The whole falling-sand game in one MonoBehaviour. Drop it on an empty GameObject and press Play:
    /// it builds a camera, a <see cref="FallingSandWorld"/> (NumSharp physics), a <see cref="SandRenderer"/>
    /// and a <see cref="SandHud"/> from code — no scene, prefab, sprite or material asset. Each frame it
    /// reads input (paint, pour, pick material, place faucets), advances the NumSharp cellular automaton,
    /// and blits the grid to the screen as a texture.
    /// </summary>
    /// <remarks>
    /// Everything physical happens inside NumSharp array math in <see cref="PowderGrid"/>; this class is
    /// purely input, stepping cadence, and drawing. Uses the legacy Input Manager and IMGUI so it needs no
    /// project setup (see the sample README for the one runtime requirement, a JIT-capable .NET backend).
    /// </remarks>
    [DisallowMultipleComponent]
    public sealed class FallingSandGame : MonoBehaviour
    {
        [Header("Grid")]
        [Tooltip("Simulation resolution in cells. 1000×500 is a high-detail default; each step is pure-vectorized O(cells) NumSharp (~31 ms at 1000×500), so drop this if you want a higher frame rate.")]
        public int GridWidth = 1000;
        [Tooltip("Simulation height in cells.")]
        public int GridHeight = 500;

        [Header("Simulation")]
        [Tooltip("Physics steps run per rendered frame. 1 keeps 1000×500 responsive (~32 fps); raise it (at lower resolutions) to settle material faster.")]
        public int SubStepsPerFrame = 1;
        [Tooltip("Random seed for reproducible fluid flow.")]
        public int Seed = 1;

        private FallingSandWorld _world;
        private SandRenderer _renderer;
        private SandHud _hud;
        private Camera _cam;

        private int _brushRadius = 4;
        private bool _paused;
        private float _fps = 60f;

        // Drag state: the last cell painted this stroke, so a fast mouse drag paints a continuous line
        // (PaintStroke) between frames instead of leaving gaps at the sampled cursor positions.
        private int _lastR, _lastC;
        private bool _hasLast;

        /// <summary>Builds the camera, world, renderer and HUD from code, and seeds an initial pile so the screen isn't blank.</summary>
        private void Start()
        {
            // Reuse the scene's camera (a fresh scene ships one) rather than adding a second; we only need
            // it to paint the letterbox background behind the world texture.
            _cam = Camera.main != null ? Camera.main : FindFirstObjectByType<Camera>();
            if (_cam == null) _cam = new GameObject("Sand Camera").AddComponent<Camera>();
            _cam.clearFlags = CameraClearFlags.SolidColor;
            _cam.backgroundColor = new Color(0.02f, 0.02f, 0.03f);
            if (_cam.GetComponent<AudioListener>() == null) _cam.gameObject.AddComponent<AudioListener>();

            _world = new FallingSandWorld(GridHeight, GridWidth, Seed, withBoundary: true);
            _world.BrushMaterial = Cell.Sand;
            _renderer = new SandRenderer(GridHeight, GridWidth);
            _hud = new SandHud();

            SeedDemoScene();
        }

        /// <summary>Opens on the Waterfall template ("water flowing down") so the world greets the player with a pretty, animated scene instead of a blank basin. Any template icon swaps it out.</summary>
        private void SeedDemoScene()
        {
            _world.LoadTemplate(0);   // 0 = Waterfall
            _world.BrushMaterial = Cell.Sand;   // restore a sensible default brush after the template's own edits
            _world.BrushRadius = _brushRadius;
        }

        /// <summary>Per-frame: input, then advance the simulation, then upload the grid to the render texture.</summary>
        private void Update()
        {
            HandleInput();
            if (!_paused) _world.Step(SubStepsPerFrame);
            _renderer.Render(_world.Grid.Snapshot());
            _fps = Mathf.Lerp(_fps, 1f / Mathf.Max(Time.unscaledDeltaTime, 1e-5f), 0.1f);
        }

        /// <summary>Reads the keyboard and mouse: material picking, brush/step tuning, painting/pouring, and faucet placement.</summary>
        private void HandleInput()
        {
            // Material selection (1-8): the classic four, then fire and lava, then wall and the eraser.
            if (Input.GetKeyDown(KeyCode.Alpha1)) _world.BrushMaterial = Cell.Sand;
            if (Input.GetKeyDown(KeyCode.Alpha2)) _world.BrushMaterial = Cell.Water;
            if (Input.GetKeyDown(KeyCode.Alpha3)) _world.BrushMaterial = Cell.Oil;
            if (Input.GetKeyDown(KeyCode.Alpha4)) _world.BrushMaterial = Cell.Smoke;
            if (Input.GetKeyDown(KeyCode.Alpha5)) _world.BrushMaterial = Cell.Fire;
            if (Input.GetKeyDown(KeyCode.Alpha6)) _world.BrushMaterial = Cell.Lava;
            if (Input.GetKeyDown(KeyCode.Alpha7)) _world.BrushMaterial = Cell.Wall;
            if (Input.GetKeyDown(KeyCode.Alpha8)) _world.BrushMaterial = Cell.Empty;   // eraser

            if (Input.GetKeyDown(KeyCode.LeftBracket)) _brushRadius = Mathf.Max(0, _brushRadius - 1);
            if (Input.GetKeyDown(KeyCode.RightBracket)) _brushRadius = Mathf.Min(40, _brushRadius + 1);
            _world.BrushRadius = _brushRadius;

            // B toggles the brush footprint (disk for pouring, square for drawing crisp walls).
            if (Input.GetKeyDown(KeyCode.B))
                _world.BrushShape = _world.BrushShape == BrushShape.Disk ? BrushShape.Square : BrushShape.Disk;

            if (Input.GetKeyDown(KeyCode.Comma)) SubStepsPerFrame = Mathf.Max(1, SubStepsPerFrame - 1);
            if (Input.GetKeyDown(KeyCode.Period)) SubStepsPerFrame = Mathf.Min(10, SubStepsPerFrame + 1);
            if (Input.GetKeyDown(KeyCode.Space)) _paused = !_paused;

            // C clears loose material (keeps walls); Shift+C wipes everything.
            if (Input.GetKeyDown(KeyCode.C))
                _world.Clear(keepWalls: !(Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift)));

            // F1..F6 load a presaved template scene (waterfall, hourglass, volcano, fountain, rain, sandbox).
            for (int t = 0; t < FallingSandWorld.TemplateCount; t++)
                if (Input.GetKeyDown(KeyCode.F1 + t)) _world.LoadTemplate(t);

            // Faucets: E places one of the current material at the cursor; F removes them all.
            if (Input.GetKeyDown(KeyCode.E) && TryCellAtMouse(out int er, out int ec))
                _world.AddEmitter(er, ec, _world.BrushMaterial);
            if (Input.GetKeyDown(KeyCode.F)) _world.ClearEmitters();

            // Left-drag paints / pours; a drag interpolates a continuous stroke between the two sampled cells
            // so a fast swipe leaves an unbroken line rather than dots.
            if (Input.GetMouseButton(0) && !IsOverHud(Input.mousePosition) && TryCellAtMouse(out int r, out int c))
            {
                if (_hasLast) _world.PaintStroke(_lastR, _lastC, r, c);
                else _world.Paint(r, c);
                _lastR = r; _lastC = c; _hasLast = true;
            }
            if (Input.GetMouseButtonUp(0)) _hasLast = false;   // end the stroke so the next drag starts fresh
        }

        /// <summary>Draws the world texture (aspect-fit) and the HUD on top.</summary>
        private void OnGUI()
        {
            if (_renderer == null) return;
            Rect rect = ComputeDrawRect();
            GUI.DrawTexture(rect, _renderer.Texture);

            var result = _hud.Draw(_world, _world.BrushMaterial, _brushRadius, _world.BrushShape, SubStepsPerFrame, _paused, _fps);
            _world.BrushMaterial = result.SelectedMaterial;      // palette click overrides
            _brushRadius = result.BrushRadius;                   // +/- buttons override
            _world.BrushRadius = _brushRadius;
            _world.BrushShape = result.BrushShape;               // shape toggle overrides
            if (result.ClearRequested) _world.Clear(keepWalls: true);
            if (result.ClearAllRequested) _world.Clear(keepWalls: false);
            if (result.TemplateRequested >= 0) _world.LoadTemplate(result.TemplateRequested);   // icon click loads a scene
        }

        /// <summary>Destroys the runtime render texture on teardown.</summary>
        private void OnDestroy() => _renderer?.Dispose();

        /// <summary>
        /// The screen rectangle the world texture is drawn into: the grid's aspect ratio fit and centred in
        /// the window (letterboxed by the camera's clear colour). Shared by drawing and input so the mouse
        /// maps to the same pixels that are shown.
        /// </summary>
        /// <returns>The aspect-fit draw rectangle in GUI (top-left origin) coordinates.</returns>
        private Rect ComputeDrawRect()
        {
            float sw = Screen.width, sh = Screen.height;
            float aspect = (float)GridWidth / GridHeight;
            float rw = sw, rh = sw / aspect;
            if (rh > sh) { rh = sh; rw = sh * aspect; }   // clamp to the shorter axis
            return new Rect((sw - rw) * 0.5f, (sh - rh) * 0.5f, rw, rh);
        }

        /// <summary>
        /// Maps the current mouse position to a grid cell, if it is over the drawn world. Converts the
        /// input system's bottom-left origin to the GUI's top-left origin so row 0 lands at the top.
        /// </summary>
        /// <param name="row">Receives the grid row (0 = top).</param>
        /// <param name="col">Receives the grid column (0 = left).</param>
        /// <returns>True if the cursor is over the world; false otherwise.</returns>
        private bool TryCellAtMouse(out int row, out int col)
        {
            row = col = 0;
            Rect rect = ComputeDrawRect();
            float mx = Input.mousePosition.x;
            float myGui = Screen.height - Input.mousePosition.y;   // flip to top-left origin
            if (mx < rect.x || mx >= rect.x + rect.width || myGui < rect.y || myGui >= rect.y + rect.height)
                return false;
            float u = (mx - rect.x) / rect.width;
            float v = (myGui - rect.y) / rect.height;              // 0 at top → grid row 0 at top
            col = Mathf.Clamp((int)(u * GridWidth), 0, GridWidth - 1);
            row = Mathf.Clamp((int)(v * GridHeight), 0, GridHeight - 1);
            return true;
        }

        /// <summary>Whether the cursor is over a HUD panel (so a click shouldn't also paint into the world).</summary>
        /// <param name="mouse">Mouse position in input coordinates (bottom-left origin).</param>
        /// <returns>True over the left palette strip, the top-right stats panel, or the bottom template strip.</returns>
        private bool IsOverHud(Vector3 mouse)
        {
            float guiY = Screen.height - mouse.y;
            bool overPalette = mouse.x < SandHud.PaletteWidth;
            bool overStats = guiY < 160f && mouse.x > Screen.width - 620f;
            bool overTemplates = guiY > Screen.height - SandHud.TemplateStripHeight;
            return overPalette || overStats || overTemplates;
        }
    }
}
