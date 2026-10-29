using System;
using NumSharp;
using NumSharp.Backends.Iteration;
using static NumSharp.Backends.Iteration.NDExpr;

namespace NumSharp.Examples.MaterialLab.Simulation
{
    /// <summary>
    /// The grid-velocity update of each substep, written as five fused NumSharp expressions over the node
    /// lanes: momentum → velocity, gravity (any direction — the tank can be tilted), the mouse "stir" force
    /// field, and collision with the obstacle distance field including Coulomb wall friction.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Why two scratch planes.</b> The collision step needs BOTH velocity components to project out the
    /// normal part, but the lanes it writes are the same lanes the momentum lives in. So velocity-before-
    /// collision goes to <c>V1X</c>/<c>V1Y</c> first, and the final two expressions read only those and the
    /// mass lane — never a lane they overwrite.
    /// </para>
    /// <para>
    /// <b>Separating boundaries.</b> Only velocity INTO a wall (vₙ &lt; 0) is removed, so material may leave a
    /// wall freely (splashes lift off the floor); friction removes tangential speed in proportion to the
    /// normal speed that was cancelled (Coulomb), which is what keeps a sand pile from sliding flat while
    /// letting water run along a ramp.
    /// </para>
    /// <para>
    /// Every value that changes at runtime (Δt, gravity, the stir) is a 0-d parameter array, so moving the
    /// mouse never recompiles a kernel.
    /// </para>
    /// <para>
    /// <b>The friction factor has its own plane.</b> Both collision kernels need the same Coulomb factor and a
    /// fused tree does not share subexpressions, so it is computed once into <c>FR</c> and read by both. Its
    /// tangential speed is <c>Sqrt(tx² + ty²)</c>, not <c>Hypot</c>: grid speeds are far from float overflow, and
    /// the overflow-safe <c>Hypot</c> measured ~19× slower (232 µs vs 12 µs for the 20,736-node plane). Measured
    /// interleaved on a settled scene, the whole update went 141 µs → 93 µs; hoisting the <c>Hypot</c> form
    /// instead was 2.2× SLOWER than leaving it inline, because inside the <c>Where</c> it only ran at wall nodes.
    /// </para>
    /// </remarks>
    public sealed class GridUpdate : IDisposable
    {
        /// <summary>The grid whose node lanes the kernels read and write.</summary>
        private readonly MpmGrid _grid;
        /// <summary>Scratch planes: velocity before collision (x, y) and the Coulomb friction factor.</summary>
        private readonly NDArray _v1x, _v1y, _fr;
        /// <summary>The five kernels: velocity + forces (x, y), friction factor, collision (x, y).</summary>
        private readonly CompiledExpression _k1x, _k1y, _kfr, _k2x, _k2y;
        /// <summary>Strided views of node lanes 1–2 (momentum in, velocity out).</summary>
        private readonly NDArray _lane1, _lane2;

        /// <summary>Gravity x (units/s²), a 0-d parameter.</summary>
        public NDArray GravityX { get; } = NDArray.Scalar(0f);
        /// <summary>Gravity y (units/s²), a 0-d parameter.</summary>
        public NDArray GravityY { get; } = NDArray.Scalar(-100f);
        /// <summary>Stir center x.</summary>
        public NDArray StirX { get; } = NDArray.Scalar(0f);
        /// <summary>Stir center y.</summary>
        public NDArray StirY { get; } = NDArray.Scalar(0f);
        /// <summary>Stir target velocity x (the mouse drag velocity).</summary>
        public NDArray StirVX { get; } = NDArray.Scalar(0f);
        /// <summary>Stir target velocity y.</summary>
        public NDArray StirVY { get; } = NDArray.Scalar(0f);
        /// <summary>1/R² of the stir disk.</summary>
        public NDArray StirInvR2 { get; } = NDArray.Scalar(1f);
        /// <summary>Per-substep blend factor toward the stir velocity at the stir center (0 = off).</summary>
        public NDArray StirStrength { get; } = NDArray.Scalar(0f);
        /// <summary>Coulomb friction coefficient of every wall.</summary>
        public NDArray Friction { get; } = NDArray.Scalar(0.5f);

        /// <summary>Compiles the five kernels over <paramref name="grid"/>'s lanes.</summary>
        /// <param name="grid">The grid (its lane views are captured).</param>
        /// <param name="dt">The shared Δt parameter (from <see cref="SolverParams.Dt"/>).</param>
        public GridUpdate(MpmGrid grid, NDArray dt)
        {
            _grid = grid;
            _v1x = np.zeros(new Shape(grid.Ny, grid.Nx), NPTypeCode.Single);
            _v1y = np.zeros(new Shape(grid.Ny, grid.Nx), NPTypeCode.Single);
            _fr = np.zeros(new Shape(grid.Ny, grid.Nx), NPTypeCode.Single);
            var lane0 = grid.Nodes[":, :, 0"];
            _lane1 = grid.Nodes[":, :, 1"];
            _lane2 = grid.Nodes[":, :, 2"];

            NDExpr m = Arr(lane0), px = Arr(_lane1), py = Arr(_lane2);
            NDExpr inv = 1.0 / Maximum(m, 1e-30);
            NDExpr h = Arr(dt);
            NDExpr dxs = Arr(grid.NodeX) - Arr(StirX), dys = Arr(grid.NodeY) - Arr(StirY);
            NDExpr stir = Arr(StirStrength) * Maximum(1.0 - (dxs * dxs + dys * dys) * Arr(StirInvR2), 0.0);
            NDExpr v0x = px * inv + h * Arr(GravityX);
            NDExpr v0y = py * inv + h * Arr(GravityY);
            _k1x = (v0x + stir * (Arr(StirVX) - v0x)).Compile();
            _k1y = (v0y + stir * (Arr(StirVY) - v0y)).Compile();

            NDExpr vx = Arr(_v1x), vy = Arr(_v1y), nx = Arr(grid.NormalX), ny = Arr(grid.NormalY);
            NDExpr vn = vx * nx + vy * ny;
            NDExpr tx = vx - vn * nx, ty = vy - vn * ny;
            // Coulomb: keep (1 + μ·vₙ/|v_t|)⁺ of the tangential velocity (vₙ < 0 at a hit, so this shrinks it).
            _kfr = Maximum(1.0 + Arr(Friction) * vn / Maximum(Sqrt(tx * tx + ty * ty), 1e-12), 0.0).Compile();
            NDExpr fr = Arr(_fr);
            NDExpr hits = LogicalAnd(Arr(grid.Phi) < 1.0, vn < 0.0);   // near a wall AND moving into it
            NDExpr has = m > 0.0;
            _k2x = Where(has, Where(hits, tx * fr, vx), 0.0).Compile();
            _k2y = Where(has, Where(hits, ty * fr, vy), 0.0).Compile();
        }

        /// <summary>
        /// Converts the scattered momentum to velocity and applies forces and boundaries, in place on the
        /// grid's lanes 1–2 (the other lanes — mass, volume, coolant, wall volume — are left as scattered).
        /// </summary>
        public void Apply()
        {
            _k1x.Evaluate(_v1x);
            _k1y.Evaluate(_v1y);
            _kfr.Evaluate(_fr);
            _k2x.Evaluate(_lane1);
            _k2y.Evaluate(_lane2);
        }

        /// <summary>Sets the stir force field for the next substeps (strength 0 turns it off).</summary>
        /// <param name="x">Center x.</param>
        /// <param name="y">Center y.</param>
        /// <param name="vx">Target velocity x.</param>
        /// <param name="vy">Target velocity y.</param>
        /// <param name="radius">Disk radius (units).</param>
        /// <param name="strength">Per-substep blend factor at the center (0..1).</param>
        public void SetStir(float x, float y, float vx, float vy, float radius, float strength)
        {
            StirX.SetValue(x); StirY.SetValue(y);
            StirVX.SetValue(vx); StirVY.SetValue(vy);
            StirInvR2.SetValue(1f / MathF.Max(radius * radius, 1e-8f));
            StirStrength.SetValue(strength);
        }

        /// <summary>Releases the scratch planes and parameters.</summary>
        public void Dispose()
        {
            _v1x.Dispose(); _v1y.Dispose(); _fr.Dispose();
            GravityX.Dispose(); GravityY.Dispose(); StirX.Dispose(); StirY.Dispose(); StirVX.Dispose(); StirVY.Dispose();
            StirInvR2.Dispose(); StirStrength.Dispose(); Friction.Dispose();
        }
    }
}
