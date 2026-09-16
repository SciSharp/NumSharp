using System;

namespace NumSharp.Examples.GravitySandbox.Physics
{
    /// <summary>
    /// Per-body PRESENTATION data — everything the view needs to draw a body that the physics does not
    /// care about. It is kept index-aligned with the rows of <see cref="NBodyState"/> and out of the
    /// physics arrays deliberately: the solver only ever touches position, velocity and mass, so colour
    /// and name never pollute the numeric hot path. <see cref="DisplayRadius"/> is the one field that
    /// also carries physical meaning — it doubles as the body's collision radius when merging is on.
    /// </summary>
    public struct BodyDescriptor
    {
        /// <summary>A human-readable label (e.g. "Earth"), shown in the HUD and used for logging.</summary>
        public string Name;
        /// <summary>Red channel of the body's colour, 0..1.</summary>
        public double ColorR;
        /// <summary>Green channel of the body's colour, 0..1.</summary>
        public double ColorG;
        /// <summary>Blue channel of the body's colour, 0..1.</summary>
        public double ColorB;
        /// <summary>
        /// The rendered sphere radius in scenario distance units. Also the collision radius: two bodies
        /// merge when their centres approach within the sum of their display radii (if collisions are on),
        /// which is why a "sun" is drawn large enough to sweep up things that fall into it.
        /// </summary>
        public double DisplayRadius;
        /// <summary>
        /// Whether this body is a self-luminous star. The view uses it to add a point light and an
        /// emissive material so the body lights the others rather than needing to be lit.
        /// </summary>
        public bool IsStar;

        /// <summary>Convenience constructor for the common case.</summary>
        /// <param name="name">Display name.</param>
        /// <param name="r">Colour red, 0..1.</param>
        /// <param name="g">Colour green, 0..1.</param>
        /// <param name="b">Colour blue, 0..1.</param>
        /// <param name="displayRadius">Rendered / collision radius in scenario units.</param>
        /// <param name="isStar">True for a self-luminous body.</param>
        public BodyDescriptor(string name, double r, double g, double b, double displayRadius, bool isStar = false)
        {
            Name = name; ColorR = r; ColorG = g; ColorB = b; DisplayRadius = displayRadius; IsStar = isStar;
        }
    }

    /// <summary>
    /// A complete, self-contained description of a simulation to run: the initial bodies, the physical
    /// constants and unit system they were authored in, the numerical scheme, and a few view hints. A
    /// scenario is a pure DATA recipe — handing one to an <see cref="NBodyWorld"/> is what starts a
    /// simulation, and a world can be reset back to its scenario at any time.
    /// </summary>
    /// <remarks>
    /// <b>Units are per-scenario and must be internally consistent.</b> <see cref="G"/>, the masses in
    /// <see cref="InitialState"/>, the distances and <see cref="TimeStep"/> all share one unit system
    /// (for example AU / solar-mass / year with <c>G = 4π²</c>, or the dimensionless <c>G = 1</c> of the
    /// figure-eight). Mixing units does not throw — it silently rescales time — so each factory in
    /// <see cref="ScenarioLibrary"/> documents the units it chose.
    /// </remarks>
    public sealed class Scenario
    {
        /// <summary>Short title shown in the scenario picker.</summary>
        public string Name { get; set; } = "Untitled";

        /// <summary>One-line explanation of what the scenario demonstrates.</summary>
        public string Description { get; set; } = "";

        /// <summary>The initial bodies. A world takes a <see cref="NBodyState.Clone"/> of this so reset can restore it.</summary>
        public NBodyState InitialState { get; set; }

        /// <summary>Presentation data, index-aligned with the rows of <see cref="InitialState"/>.</summary>
        public BodyDescriptor[] Descriptors { get; set; } = Array.Empty<BodyDescriptor>();

        /// <summary>Gravitational constant in this scenario's unit system.</summary>
        public double G { get; set; } = 1.0;

        /// <summary>Plummer softening length ε in this scenario's distance unit.</summary>
        public double Softening { get; set; } = 1e-3;

        /// <summary>The integrator time step in this scenario's time unit. Smaller is more accurate and slower.</summary>
        public double TimeStep { get; set; } = 1e-3;

        /// <summary>
        /// How much simulated time elapses per real second at 1× speed. Lets a slow orbit (a real planet)
        /// and a fast one (a tight binary) both be watchable without re-authoring the physics.
        /// </summary>
        public double TimeScale { get; set; } = 1.0;

        /// <summary>The force law (Newtonian by default; relativistic for the precession demo).</summary>
        public ForceModel Model { get; set; } = ForceModel.Newtonian;

        /// <summary>Speed of light in scenario units; used only by <see cref="ForceModel.RelativisticPrecession"/>.</summary>
        public double SpeedOfLight { get; set; } = double.PositiveInfinity;

        /// <summary>The integration scheme (leapfrog by default).</summary>
        public IntegrationMethod Method { get; set; } = IntegrationMethod.Leapfrog;

        /// <summary>Whether bodies merge (perfectly inelastically, conserving momentum) when they touch.</summary>
        public bool EnableCollisions { get; set; } = false;

        /// <summary>A view hint: how far back the camera should start, in scenario distance units.</summary>
        public double CameraDistance { get; set; } = 10.0;

        /// <summary>A view hint: how many trail points to keep per body (0 disables trails).</summary>
        public int TrailLength { get; set; } = 200;
    }
}
