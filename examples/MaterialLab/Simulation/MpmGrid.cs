using System;
using NumSharp;

namespace NumSharp.Examples.MaterialLab.Simulation
{
    /// <summary>
    /// The Eulerian background grid of the MPM solver plus the static obstacle field.
    /// <list type="bullet">
    /// <item><description><see cref="Nodes"/> — an <c>(Ny, Nx, 8)</c> float32 array, interleaved per node as
    /// <c>[mass, momentum-x, momentum-y, volume, coolant-mass, wall-volume, –, –]</c> (see the <c>Lane*</c>
    /// constants); after the grid update lanes 1–2 hold velocity. Interleaving is what lets the transfer kernels
    /// move a whole node with one 256-bit load/FMA/store; the volume lane gives the local mixture density
    /// (mass/volume) that drives buoyant separation and, with the wall-volume lane, the occupancy that sets a
    /// liquid's pressure.</description></item>
    /// <item><description>The obstacle field — a signed distance <see cref="Phi"/> (in cells, positive in free
    /// space), its outward normals <see cref="NormalX"/>/<see cref="NormalY"/> and the derived node mask
    /// <see cref="Solid"/>, plus a 4× finer copy of φ (<see cref="PhiFine"/>) for crisp rendering. One smooth
    /// representation serves grid collisions, wall friction, particle push-out and the renderer.</description></item>
    /// </list>
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>Analytic obstacles.</b> Walls are built from exact shapes — disks, capsules (thick segments) and boxes —
    /// combined with CSG on the distance field: adding a solid is φ ← min(φ, sdf), carving free space is
    /// φ ← max(φ, −sdf). Both operations only change values near the shape, so each edit touches just the
    /// shape's bounding box (plus a band) at both resolutions: painting walls with the mouse stays cheap, and a
    /// ramp is a true straight ramp rather than a staircase of grid nodes (particles slide on it smoothly).
    /// </para>
    /// <para>
    /// φ is clamped to ±<see cref="Band"/> cells; only values near the surface matter (collisions act within a
    /// cell, rendering within a pixel or two), and the clamp keeps every edit local. Row <c>j</c> is height
    /// <c>j·Δx</c> (y up); the domain is <see cref="Ny"/> cells tall with Δx = 1/<see cref="Ny"/>.
    /// </para>
    /// </remarks>
    public sealed unsafe class MpmGrid : IDisposable
    {
        /// <summary>Wall thickness (in cells) of the tank around the domain edge.</summary>
        public const int Border = 3;

        /// <summary>Distance-field clamp (cells) — edits only reach this far from a shape.</summary>
        public const float Band = 8f;

        /// <summary>Resolution multiplier of <see cref="PhiFine"/> over the grid.</summary>
        public const int FineScale = 4;

        /// <summary>Floats per node in <see cref="Nodes"/> (one Vector256).</summary>
        public const int Lanes = 8;
        /// <summary>Node lane: mass Σw·m.</summary>
        public const int LaneMass = 0;
        /// <summary>Node lane: momentum x (velocity x after the grid update).</summary>
        public const int LaneMomX = 1;
        /// <summary>Node lane: momentum y (velocity y after the grid update).</summary>
        public const int LaneMomY = 2;
        /// <summary>Node lane: particle volume Σw·V₀ — mass/volume is the local mixture density.</summary>
        public const int LaneVolume = 3;
        /// <summary>Node lane: mass of coolant (water) Σw·m_water — how wet a place is, for lava quenching.</summary>
        public const int LaneCoolant = 4;
        /// <summary>
        /// Node lane: WALL volume — the volume the obstacle itself stands in for, Δx²·C(½ − φ) with C the quadratic
        /// B-spline's cumulative weight. Constant between obstacle edits, so <see cref="ClearNodes"/> restores it
        /// instead of zeroing it, and the transfers never write it. A particle's occupancy is its gathered volume
        /// PLUS this lane: next to a wall, the part of its kernel inside the wall then reads as full rather than as
        /// empty space. Without it every liquid packs itself denser against floors and walls until the missing
        /// kernel weight is made up — a resting pool measurably loses depth.
        /// </summary>
        /// <remarks>
        /// Why C(½ − φ): particles are kept at least ½ cell from an obstacle surface, so at rest packing a node at
        /// distance φ from a flat wall collects C(φ − ½)·Δx² of particle volume; the lane supplies the complement,
        /// C(½ − φ)·Δx², and the two sum to exactly one full cell for every node near the wall.
        /// </remarks>
        public const int LaneWall = 5;

        /// <summary>Nodes across.</summary>
        public int Nx { get; }
        /// <summary>Nodes up.</summary>
        public int Ny { get; }
        /// <summary>Node count.</summary>
        public int G => Nx * Ny;
        /// <summary>Cell size Δx = 1/Ny.</summary>
        public float Dx { get; }
        /// <summary>1/Δx.</summary>
        public float InvDx { get; }
        /// <summary>Domain width in units (Nx·Δx).</summary>
        public float Width => Nx * Dx;

        /// <summary>The interleaved node array (see the class summary).</summary>
        public NDArray Nodes { get; }
        /// <summary>Obstacle node mask, 1 = solid (φ &lt; 0).</summary>
        public NDArray Solid { get; }
        /// <summary>Signed distance to the obstacle surface in cells (positive outside), at the grid nodes.</summary>
        public NDArray Phi { get; }
        /// <summary>Outward obstacle normal, x component (zero far from any wall).</summary>
        public NDArray NormalX { get; }
        /// <summary>Outward obstacle normal, y component.</summary>
        public NDArray NormalY { get; }
        /// <summary>Node x coordinate (units), for spatial forces such as stirring.</summary>
        public NDArray NodeX { get; }
        /// <summary>Node y coordinate (units).</summary>
        public NDArray NodeY { get; }

        /// <summary>φ at <see cref="FineScale"/>× the grid resolution, <c>(Ny·4, Nx·4)</c>, still in GRID-cell units (for rendering).</summary>
        public float[] PhiFine { get; }
        /// <summary>Width of <see cref="PhiFine"/>.</summary>
        public int FineW => Nx * FineScale;
        /// <summary>Height of <see cref="PhiFine"/>.</summary>
        public int FineH => Ny * FineScale;

        /// <summary>Incremented whenever the obstacle field changes (the renderer re-uploads its wall texture).</summary>
        public int ObstacleVersion { get; private set; }

        /// <summary>
        /// The node array's cleared state: zero everywhere except <see cref="LaneWall"/>, which holds each node's
        /// wall volume. <see cref="ClearNodes"/> copies it over <see cref="Nodes"/>; <see cref="Refresh"/> keeps its
        /// wall lane in step with φ.
        /// </summary>
        private readonly float[] _blankNodes;

        /// <summary>Creates an empty tank of <paramref name="nx"/>×<paramref name="ny"/> nodes.</summary>
        /// <param name="nx">Nodes across (≥ 16).</param>
        /// <param name="ny">Nodes up (≥ 16).</param>
        /// <exception cref="ArgumentOutOfRangeException">A dimension is smaller than 16.</exception>
        public MpmGrid(int nx, int ny)
        {
            if (nx < 16) throw new ArgumentOutOfRangeException(nameof(nx));
            if (ny < 16) throw new ArgumentOutOfRangeException(nameof(ny));
            Nx = nx; Ny = ny;
            Dx = 1f / ny; InvDx = ny;
            Nodes = np.zeros(new Shape(ny, nx, Lanes), NPTypeCode.Single);
            Solid = np.zeros(new Shape(ny, nx), NPTypeCode.Single);
            Phi = np.zeros(new Shape(ny, nx), NPTypeCode.Single);
            NormalX = np.zeros(new Shape(ny, nx), NPTypeCode.Single);
            NormalY = np.zeros(new Shape(ny, nx), NPTypeCode.Single);
            // Node coordinates via NumSharp broadcasting: x varies along columns, y along rows.
            NodeX = (np.arange(nx).astype(NPTypeCode.Single) * Dx).reshape(1, nx) * np.ones(new Shape(ny, 1), NPTypeCode.Single);
            NodeY = (np.arange(ny).astype(NPTypeCode.Single) * Dx).reshape(ny, 1) * np.ones(new Shape(1, nx), NPTypeCode.Single);
            PhiFine = new float[FineW * FineH];
            _blankNodes = new float[ny * nx * Lanes];
            ResetObstacles();
        }

        // ------------------------------------------------------------------ shapes (grid-cell coordinates)

        /// <summary>Signed distance of the tank's free interior (positive inside the tank, negative in its walls), in cells.</summary>
        private float TankSdf(float x, float y)
        {
            float b = Border - 0.5f;   // the wall surface lies half a cell outside the last wall node
            return MathF.Min(MathF.Min(x - b, (Nx - 1 - b) - x), MathF.Min(y - b, (Ny - 1 - b) - y));
        }

        /// <summary>A solid shape in grid-cell coordinates: its exact signed distance and its bounding box (which bounds every edit).</summary>
        private abstract class ShapeSdf
        {
            /// <summary>Bounding box, left.</summary>
            public float MinX;
            /// <summary>Bounding box, bottom.</summary>
            public float MinY;
            /// <summary>Bounding box, right.</summary>
            public float MaxX;
            /// <summary>Bounding box, top.</summary>
            public float MaxY;

            /// <summary>Signed distance to the shape surface (negative inside), in cells.</summary>
            /// <param name="x">X in cells.</param>
            /// <param name="y">Y in cells.</param>
            /// <returns>The distance.</returns>
            public abstract float Sdf(float x, float y);
        }

        /// <summary>A disk.</summary>
        private sealed class CircleSdf : ShapeSdf
        {
            /// <summary>Center and radius (cells).</summary>
            private readonly float _cx, _cy, _r;

            /// <summary>Creates the disk.</summary>
            /// <param name="cx">Center x (cells).</param>
            /// <param name="cy">Center y (cells).</param>
            /// <param name="r">Radius (cells).</param>
            public CircleSdf(float cx, float cy, float r) { _cx = cx; _cy = cy; _r = r; MinX = cx - r; MaxX = cx + r; MinY = cy - r; MaxY = cy + r; }

            /// <inheritdoc/>
            public override float Sdf(float x, float y) => MathF.Sqrt((x - _cx) * (x - _cx) + (y - _cy) * (y - _cy)) - _r;
        }

        /// <summary>A capsule — every point within <c>r</c> of a segment (a thick line with round caps).</summary>
        private sealed class CapsuleSdf : ShapeSdf
        {
            /// <summary>Segment endpoints, radius (cells) and squared length.</summary>
            private readonly float _ax, _ay, _bx, _by, _r, _len2;

            /// <summary>Creates the capsule.</summary>
            /// <param name="ax">Start x (cells).</param>
            /// <param name="ay">Start y (cells).</param>
            /// <param name="bx">End x (cells).</param>
            /// <param name="by">End y (cells).</param>
            /// <param name="r">Radius (cells).</param>
            public CapsuleSdf(float ax, float ay, float bx, float by, float r)
            {
                _ax = ax; _ay = ay; _bx = bx; _by = by; _r = r;
                _len2 = MathF.Max((bx - ax) * (bx - ax) + (by - ay) * (by - ay), 1e-12f);
                MinX = MathF.Min(ax, bx) - r; MaxX = MathF.Max(ax, bx) + r; MinY = MathF.Min(ay, by) - r; MaxY = MathF.Max(ay, by) + r;
            }

            /// <inheritdoc/>
            public override float Sdf(float x, float y)
            {
                float u = Math.Clamp(((x - _ax) * (_bx - _ax) + (y - _ay) * (_by - _ay)) / _len2, 0f, 1f);
                float px = _ax + u * (_bx - _ax) - x, py = _ay + u * (_by - _ay) - y;
                return MathF.Sqrt(px * px + py * py) - _r;
            }
        }

        /// <summary>An axis-aligned box with rounded corners.</summary>
        private sealed class BoxSdf : ShapeSdf
        {
            /// <summary>Center, half extents and corner radius (cells).</summary>
            private readonly float _cx, _cy, _hx, _hy, _round;

            /// <summary>Creates the box.</summary>
            /// <param name="x0">One corner x (cells).</param>
            /// <param name="y0">One corner y (cells).</param>
            /// <param name="x1">Opposite corner x (cells).</param>
            /// <param name="y1">Opposite corner y (cells).</param>
            /// <param name="round">Corner radius (cells).</param>
            public BoxSdf(float x0, float y0, float x1, float y1, float round)
            {
                _cx = 0.5f * (x0 + x1); _cy = 0.5f * (y0 + y1); _hx = 0.5f * MathF.Abs(x1 - x0); _hy = 0.5f * MathF.Abs(y1 - y0); _round = round;
                MinX = MathF.Min(x0, x1); MaxX = MathF.Max(x0, x1); MinY = MathF.Min(y0, y1); MaxY = MathF.Max(y0, y1);
            }

            /// <inheritdoc/>
            public override float Sdf(float x, float y)
            {
                float dx = MathF.Abs(x - _cx) - (_hx - _round), dy = MathF.Abs(y - _cy) - (_hy - _round);
                float ox = MathF.Max(dx, 0f), oy = MathF.Max(dy, 0f);
                return MathF.Sqrt(ox * ox + oy * oy) + MathF.Min(MathF.Max(dx, dy), 0f) - _round;
            }
        }

        // ------------------------------------------------------------------ editing API (units)

        /// <summary>Removes every painted obstacle, leaving just the tank walls.</summary>
        public void ResetObstacles()
        {
            float* phi = Phi.Unsafe.Pointer<float>();
            for (int j = 0; j < Ny; j++)
                for (int i = 0; i < Nx; i++)
                    phi[j * Nx + i] = Math.Clamp(TankSdf(i, j), -Band, Band);
            float inv = 1f / FineScale;
            for (int j = 0; j < FineH; j++)
                for (int i = 0; i < FineW; i++)
                    PhiFine[j * FineW + i] = Math.Clamp(TankSdf((i + 0.5f) * inv - 0.5f, (j + 0.5f) * inv - 0.5f), -Band, Band);
            Refresh(0, 0, Nx - 1, Ny - 1);
        }

        /// <summary>Whether a node belongs to the fixed tank walls (never erasable).</summary>
        /// <param name="i">Column.</param>
        /// <param name="j">Row.</param>
        /// <returns>True inside the border band.</returns>
        public bool IsBorder(int i, int j) => i < Border || j < Border || i >= Nx - Border || j >= Ny - Border;

        /// <summary>Adds (or carves away) a disk of obstacle. The tank walls cannot be carved.</summary>
        /// <param name="cx">Center x (units).</param>
        /// <param name="cy">Center y (units).</param>
        /// <param name="radius">Radius (units).</param>
        /// <param name="solid">True to add obstacle, false to erase.</param>
        public void PaintDisk(float cx, float cy, float radius, bool solid)
            => Apply(new CircleSdf(cx * InvDx, cy * InvDx, radius * InvDx), solid);

        /// <summary>Adds a thick line segment (a capsule) — ramps, ledges, funnels, and mouse wall strokes.</summary>
        /// <param name="x0">Start x (units).</param>
        /// <param name="y0">Start y.</param>
        /// <param name="x1">End x.</param>
        /// <param name="y1">End y.</param>
        /// <param name="thickness">Half thickness (units).</param>
        /// <param name="solid">True to add, false to carve.</param>
        public void PaintSegment(float x0, float y0, float x1, float y1, float thickness, bool solid = true)
            => Apply(new CapsuleSdf(x0 * InvDx, y0 * InvDx, x1 * InvDx, y1 * InvDx, thickness * InvDx), solid);

        /// <summary>Adds an axis-aligned rectangle (slightly rounded corners, like cut stone).</summary>
        /// <param name="x0">Left (units).</param>
        /// <param name="y0">Bottom.</param>
        /// <param name="x1">Right.</param>
        /// <param name="y1">Top.</param>
        public void PaintRect(float x0, float y0, float x1, float y1)
            => Apply(new BoxSdf(x0 * InvDx, y0 * InvDx, x1 * InvDx, y1 * InvDx, 0.35f), solid: true);

        /// <summary>
        /// CSG-applies one shape to both distance fields inside its bounding box (+ band), re-imposes the tank
        /// walls after a carve, and refreshes the normals and mask there.
        /// </summary>
        /// <param name="s">The shape (grid-cell coordinates).</param>
        /// <param name="solid">Add (union) or carve (subtract).</param>
        private void Apply(ShapeSdf s, bool solid)
        {
            float* phi = Phi.Unsafe.Pointer<float>();
            int i0 = Math.Max(0, (int)MathF.Floor(s.MinX - Band)), i1 = Math.Min(Nx - 1, (int)MathF.Ceiling(s.MaxX + Band));
            int j0 = Math.Max(0, (int)MathF.Floor(s.MinY - Band)), j1 = Math.Min(Ny - 1, (int)MathF.Ceiling(s.MaxY + Band));
            if (i0 > i1 || j0 > j1) return;
            for (int j = j0; j <= j1; j++)
                for (int i = i0; i <= i1; i++)
                    phi[j * Nx + i] = Combine(phi[j * Nx + i], s.Sdf(i, j), solid, i, j);
            float inv = 1f / FineScale;
            int fi0 = i0 * FineScale, fi1 = Math.Min(FineW - 1, (i1 + 1) * FineScale), fj0 = j0 * FineScale, fj1 = Math.Min(FineH - 1, (j1 + 1) * FineScale);
            for (int j = fj0; j <= fj1; j++)
                for (int i = fi0; i <= fi1; i++)
                {
                    float x = (i + 0.5f) * inv - 0.5f, y = (j + 0.5f) * inv - 0.5f;
                    PhiFine[j * FineW + i] = Combine(PhiFine[j * FineW + i], s.Sdf(x, y), solid, x, y);
                }
            Refresh(Math.Max(0, i0 - 1), Math.Max(0, j0 - 1), Math.Min(Nx - 1, i1 + 1), Math.Min(Ny - 1, j1 + 1));
        }

        /// <summary>One CSG step: union (add) or subtraction (carve, then the tank walls are re-imposed), clamped to the band.</summary>
        private float Combine(float phi, float sdf, bool solid, float x, float y)
        {
            float v = solid ? MathF.Min(phi, sdf) : MathF.Min(MathF.Max(phi, -sdf), TankSdf(x, y));
            return Math.Clamp(v, -Band, Band);
        }

        /// <summary>
        /// Recomputes normals (central differences of φ), the node mask and the wall-volume lane of the cleared-node
        /// template over a region, then bumps the version.
        /// </summary>
        /// <param name="i0">First column.</param>
        /// <param name="j0">First row.</param>
        /// <param name="i1">Last column (inclusive).</param>
        /// <param name="j1">Last row (inclusive).</param>
        private void Refresh(int i0, int j0, int i1, int j1)
        {
            float* phi = Phi.Unsafe.Pointer<float>();
            float* nx = NormalX.Unsafe.Pointer<float>();
            float* ny = NormalY.Unsafe.Pointer<float>();
            float* solidMask = Solid.Unsafe.Pointer<float>();
            float cell = Dx * Dx;
            for (int j = j0; j <= j1; j++)
                for (int i = i0; i <= i1; i++)
                {
                    int k = j * Nx + i;
                    float gx = phi[j * Nx + Math.Min(Nx - 1, i + 1)] - phi[j * Nx + Math.Max(0, i - 1)];
                    float gy = phi[Math.Min(Ny - 1, j + 1) * Nx + i] - phi[Math.Max(0, j - 1) * Nx + i];
                    float len = MathF.Sqrt(gx * gx + gy * gy);
                    if (len < 1e-6f) { nx[k] = 0; ny[k] = 0; }
                    else { nx[k] = gx / len; ny[k] = gy / len; }
                    solidMask[k] = phi[k] < 0f ? 1f : 0f;
                    _blankNodes[k * Lanes + LaneWall] = cell * SplineCdf(0.5f - phi[k]);
                }
            ObstacleVersion++;
        }

        /// <summary>
        /// Cumulative weight of the quadratic B-spline, C(t) = ∫ N(s) ds from −∞ to t: 0 below −1.5, 1 above 1.5,
        /// and the piecewise cubic in between whose derivative is the transfer kernel. C(½ − φ) is the part of a
        /// node's support lying beyond a flat wall at distance φ − ½ (see <see cref="LaneWall"/>).
        /// </summary>
        /// <param name="t">Offset in cells.</param>
        /// <returns>The cumulative weight in [0, 1].</returns>
        private static float SplineCdf(float t)
        {
            if (t <= -1.5f) return 0f;
            if (t <= -0.5f) { float u = t + 1.5f; return u * u * u / 6f; }
            if (t < 0.5f) return 0.5f + 0.75f * t - t * t * t / 3f;
            if (t < 1.5f) { float u = 1.5f - t; return 1f - u * u * u / 6f; }
            return 1f;
        }

        /// <summary>
        /// Resets every node for a new particle-to-grid transfer: all lanes zero except <see cref="LaneWall"/>,
        /// which is restored from the obstacle field (one block copy of the template — the cost of a clear).
        /// </summary>
        public void ClearNodes() => _blankNodes.AsSpan().CopyTo(new Span<float>(Nodes.Unsafe.Pointer<float>(), G * Lanes));

        /// <summary>Releases the arrays.</summary>
        public void Dispose()
        {
            Nodes.Dispose(); Solid.Dispose(); Phi.Dispose(); NormalX.Dispose(); NormalY.Dispose(); NodeX.Dispose(); NodeY.Dispose();
        }
    }
}
