using UnityEngine;
using NumSharp.Examples.FallingSand.Simulation;

namespace NumSharp.Examples.FallingSand
{
    /// <summary>
    /// Turns the simulation grid into a pixel image each frame: one <see cref="Texture2D"/> texel per
    /// cell, coloured by a material lookup table. A falling-sand world is fundamentally a bitmap, so this
    /// is the natural (and fastest) way to draw it — a single texture upload per frame rather than a
    /// GameObject per grain. The texture is created at runtime, so the sample needs no image assets.
    /// </summary>
    /// <remarks>
    /// <b>Vertical flip.</b> The grid's row 0 is the TOP of the world, but a <see cref="Texture2D"/>'s row
    /// 0 is the BOTTOM. <see cref="Render"/> maps grid row <c>r</c> to texture row <c>Height-1-r</c> so the
    /// world draws upright, which also keeps mouse-to-cell mapping in the game a simple top-down affine map.
    /// </remarks>
    public sealed class SandRenderer
    {
        private readonly Texture2D _tex;
        private readonly Color32[] _buffer;   // reused pixel buffer (no per-frame allocation)
        private readonly Color32[] _lut;      // material id → colour
        private readonly int _height;
        private readonly int _width;

        /// <summary>The live texture; the game draws this to the screen.</summary>
        public Texture2D Texture => _tex;

        /// <summary>Creates the render texture and colour table for a grid of the given size.</summary>
        /// <param name="height">Grid height in cells.</param>
        /// <param name="width">Grid width in cells.</param>
        public SandRenderer(int height, int width)
        {
            _height = height;
            _width = width;
            _tex = new Texture2D(width, height, TextureFormat.RGBA32, false);
            _tex.filterMode = FilterMode.Point;   // crisp cells, no blur between texels
            _tex.wrapMode = TextureWrapMode.Clamp;
            _buffer = new Color32[width * height];
            _lut = BuildLut();
        }

        /// <summary>Builds the material→colour lookup table from <see cref="Cell.Color"/> (0..1 floats → 0..255 bytes).</summary>
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
        /// Uploads one frame: maps every cell id in <paramref name="snapshot"/> (a flat C-order grid) to a
        /// texel through the colour table, applying the top/bottom flip, then pushes the buffer to the GPU.
        /// </summary>
        /// <param name="snapshot">The grid as <c>int[Height*Width]</c> in row-major order (from <see cref="PowderGrid.Snapshot"/>).</param>
        public void Render(int[] snapshot)
        {
            for (int r = 0; r < _height; r++)
            {
                int texRow = _height - 1 - r;         // flip: grid top → texture top
                int src = r * _width;
                int dst = texRow * _width;
                for (int c = 0; c < _width; c++)
                {
                    int id = snapshot[src + c];
                    // Clamp defensively so a stray out-of-range id can never index past the table.
                    _buffer[dst + c] = _lut[(uint)id < (uint)_lut.Length ? id : 0];
                }
            }
            _tex.SetPixels32(_buffer);
            _tex.Apply(false);   // false: don't recompute mipmaps (there are none)
        }

        /// <summary>Destroys the runtime texture. Call on teardown to avoid leaking it for the play session.</summary>
        public void Dispose()
        {
            if (_tex != null) Object.Destroy(_tex);
        }
    }
}
