using System;
using NumSharp.Examples.MaterialLab.App.Native;

namespace NumSharp.Examples.MaterialLab.App.Rendering
{
    /// <summary>A straight-alpha RGBA color in 0..1.</summary>
    /// <param name="R">Red.</param>
    /// <param name="G">Green.</param>
    /// <param name="B">Blue.</param>
    /// <param name="A">Alpha (opacity).</param>
    internal readonly record struct Rgba(float R, float G, float B, float A = 1f)
    {
        /// <summary>Returns the same color with a different alpha.</summary>
        /// <param name="a">New alpha.</param>
        /// <returns>The color.</returns>
        public Rgba WithAlpha(float a) => new Rgba(R, G, B, a);

        /// <summary>Builds a color from 0–255 components (as designers write them).</summary>
        /// <param name="r">Red 0–255.</param>
        /// <param name="g">Green 0–255.</param>
        /// <param name="b">Blue 0–255.</param>
        /// <param name="a">Alpha 0–1.</param>
        /// <returns>The color.</returns>
        public static Rgba Bytes(int r, int g, int b, float a = 1f) => new Rgba(r / 255f, g / 255f, b / 255f, a);
    }

    /// <summary>
    /// Immediate-mode 2-D overlay for the HUD: queue rectangles and text during a frame, then
    /// <see cref="End"/> uploads every quad as an instance and draws them in ONE call over the frame.
    /// Coordinates are pixels with the origin at the top-left, y down.
    /// </summary>
    /// <remarks>
    /// All primitives sample the shared <see cref="FontAtlas"/> texture — flat fills sample its solid white
    /// block — so ordering is just submission order and there is never a texture switch.
    /// </remarks>
    internal sealed unsafe class Overlay : IDisposable
    {
        /// <summary>Instance stride in floats: rectangle (x, y, w, h), UV rectangle, color.</summary>
        private const int FloatsPerQuad = 12;   // rect(4) uv(4) color(4)

        /// <summary>Expands a unit quad into each instance's pixel rectangle (y down) and interpolates its UVs.</summary>
        private const string VertexSrc = @"#version 330 core
layout(location = 0) in vec2 aCorner;
layout(location = 1) in vec4 aRect;
layout(location = 2) in vec4 aUv;
layout(location = 3) in vec4 aColor;
uniform vec2 uViewport;
out vec2 vUv;
out vec4 vColor;
void main() {
    vec2 p = aRect.xy + aCorner * aRect.zw;
    gl_Position = vec4(p.x / uViewport.x * 2.0 - 1.0, 1.0 - p.y / uViewport.y * 2.0, 0.0, 1.0);
    vUv = mix(aUv.xy, aUv.zw, aCorner);
    vColor = aColor;
}";

        /// <summary>Tints the atlas coverage (the red channel) with the instance color.</summary>
        private const string FragmentSrc = @"#version 330 core
in vec2 vUv;
in vec4 vColor;
uniform sampler2D uAtlas;
out vec4 oColor;
void main() {
    float coverage = texture(uAtlas, vUv).r;
    oColor = vec4(vColor.rgb, vColor.a * coverage);
}";

        /// <summary>The overlay shader program.</summary>
        private readonly GlProgram _program;
        /// <summary>The atlas every primitive samples.</summary>
        private readonly FontAtlas _atlas;
        /// <summary>Vertex array, the shared unit-quad buffer, and the per-frame instance buffer.</summary>
        private uint _vao, _quadVbo, _instVbo;
        /// <summary>CPU staging of this frame's instances (grows on demand).</summary>
        private float[] _data = new float[FloatsPerQuad * 4096];
        /// <summary>Instances staged since <see cref="Begin"/>.</summary>
        private int _count;
        /// <summary>Current size of the instance buffer on the GPU (it is re-specified — orphaned — each frame, grown when needed).</summary>
        private int _capacityBytes;
        /// <summary>Viewport size of the current frame in pixels.</summary>
        private int _vw, _vh;

        /// <summary>The atlas (fonts) this overlay draws with.</summary>
        public FontAtlas Atlas => _atlas;

        /// <summary>Creates the overlay pipeline. Requires a current GL context.</summary>
        /// <param name="atlas">The font atlas (owned by the caller).</param>
        public Overlay(FontAtlas atlas)
        {
            _atlas = atlas;
            _program = new GlProgram("overlay", VertexSrc, FragmentSrc, "aCorner", "aRect", "aUv", "aColor");
            uint vao, qvbo, ivbo;
            Gl.GenVertexArrays(1, &vao);
            Gl.GenBuffers(1, &qvbo);
            Gl.GenBuffers(1, &ivbo);
            _vao = vao; _quadVbo = qvbo; _instVbo = ivbo;
            Gl.BindVertexArray(_vao);
            float* corners = stackalloc float[] { 0, 0, 1, 0, 0, 1, 1, 1 };
            Gl.BindBuffer(Gl.ARRAY_BUFFER, _quadVbo);
            Gl.BufferData(Gl.ARRAY_BUFFER, 8 * sizeof(float), corners, Gl.STATIC_DRAW);
            Gl.EnableVertexAttribArray(0);
            Gl.VertexAttribPointer(0, 2, Gl.FLOAT, 0, 2 * sizeof(float), 0);
            Gl.BindBuffer(Gl.ARRAY_BUFFER, _instVbo);
            int stride = FloatsPerQuad * sizeof(float);
            for (uint a = 1; a <= 3; a++)
            {
                Gl.EnableVertexAttribArray(a);
                Gl.VertexAttribPointer(a, 4, Gl.FLOAT, 0, stride, (nint)((a - 1) * 4 * sizeof(float)));
                Gl.VertexAttribDivisor(a, 1);
            }
            Gl.BindVertexArray(0);
        }

        /// <summary>Starts a frame's overlay for a target of the given pixel size (discarding anything queued).</summary>
        /// <param name="viewportWidth">Target width in pixels.</param>
        /// <param name="viewportHeight">Target height in pixels.</param>
        public void Begin(int viewportWidth, int viewportHeight)
        {
            _vw = viewportWidth;
            _vh = viewportHeight;
            _count = 0;
        }

        /// <summary>Queues a flat rectangle.</summary>
        /// <param name="x">Left (pixels).</param>
        /// <param name="y">Top (pixels).</param>
        /// <param name="w">Width.</param>
        /// <param name="h">Height.</param>
        /// <param name="c">Fill color.</param>
        public void Rect(float x, float y, float w, float h, Rgba c)
        {
            var (u, v) = _atlas.White;
            Push(x, y, w, h, u, v, u, v, c);
        }

        /// <summary>Queues a 1-pixel rectangle outline.</summary>
        /// <param name="x">Left.</param>
        /// <param name="y">Top.</param>
        /// <param name="w">Width.</param>
        /// <param name="h">Height.</param>
        /// <param name="c">Line color.</param>
        /// <param name="t">Line thickness in pixels.</param>
        public void Frame(float x, float y, float w, float h, Rgba c, float t = 1f)
        {
            Rect(x, y, w, t, c);
            Rect(x, y + h - t, w, t, c);
            Rect(x, y + t, t, h - 2 * t, c);
            Rect(x + w - t, y + t, t, h - 2 * t, c);
        }

        /// <summary>
        /// Queues a text run with its top-left at (<paramref name="x"/>, <paramref name="y"/>). The pen is
        /// snapped to whole pixels so glyphs map 1:1 onto atlas texels (crisp, unblurred text).
        /// </summary>
        /// <param name="face">Font face.</param>
        /// <param name="text">Text (unknown characters render as spaces).</param>
        /// <param name="x">Left in pixels.</param>
        /// <param name="y">Top in pixels.</param>
        /// <param name="c">Text color.</param>
        /// <returns>The advance width of the run in pixels.</returns>
        public float Text(FontFace face, ReadOnlySpan<char> text, float x, float y, Rgba c)
        {
            float pen = MathF.Round(x);
            float top = MathF.Round(y);
            float start = pen;
            foreach (char ch in text)
            {
                if (!face.Glyphs.TryGetValue(ch, out var g) && !face.Glyphs.TryGetValue(' ', out g)) continue;
                if (ch != ' ')
                    Push(pen - face.Pad, top - face.Pad, g.Width, g.Height, g.U0, g.V0, g.U1, g.V1, c);
                pen += g.Advance;
            }
            return pen - start;
        }

        /// <summary>Queues text with a 1-pixel dark drop shadow, readable over any simulation background.</summary>
        /// <param name="face">Font face.</param>
        /// <param name="text">Text.</param>
        /// <param name="x">Left.</param>
        /// <param name="y">Top.</param>
        /// <param name="c">Text color.</param>
        /// <returns>The advance width.</returns>
        public float ShadowText(FontFace face, ReadOnlySpan<char> text, float x, float y, Rgba c)
        {
            Text(face, text, x + 1, y + 1, new Rgba(0, 0, 0, 0.75f * c.A));
            return Text(face, text, x, y, c);
        }

        /// <summary>Appends one quad instance, growing the staging array when needed.</summary>
        private void Push(float x, float y, float w, float h, float u0, float v0, float u1, float v1, Rgba c)
        {
            if ((_count + 1) * FloatsPerQuad > _data.Length) Array.Resize(ref _data, _data.Length * 2);
            int o = _count * FloatsPerQuad;
            _data[o] = x; _data[o + 1] = y; _data[o + 2] = w; _data[o + 3] = h;
            _data[o + 4] = u0; _data[o + 5] = v0; _data[o + 6] = u1; _data[o + 7] = v1;
            _data[o + 8] = c.R; _data[o + 9] = c.G; _data[o + 10] = c.B; _data[o + 11] = c.A;
            _count++;
        }

        /// <summary>
        /// Uploads and draws everything queued since <see cref="Begin"/> into the currently bound target,
        /// alpha-blended over it.
        /// </summary>
        public void End()
        {
            if (_count == 0) return;
            _program.Use();
            _program.Set("uViewport", _vw, _vh);
            _program.Set("uAtlas", 0);
            Gl.ActiveTexture(Gl.TEXTURE0);
            Gl.BindTexture(Gl.TEXTURE_2D, _atlas.Texture);
            Gl.Enable(Gl.BLEND);
            Gl.BlendEquation(Gl.FUNC_ADD);
            Gl.BlendFunc(Gl.SRC_ALPHA, Gl.ONE_MINUS_SRC_ALPHA);
            Gl.BindVertexArray(_vao);
            Gl.BindBuffer(Gl.ARRAY_BUFFER, _instVbo);
            int bytes = _count * FloatsPerQuad * sizeof(float);
            fixed (float* p = _data)
            {
                // Orphan-and-refill: a fresh store each frame, so the driver never waits for the GPU to finish reading last frame's quads.
                if (bytes > _capacityBytes) _capacityBytes = Math.Max(bytes, _capacityBytes * 2);
                Gl.BufferData(Gl.ARRAY_BUFFER, _capacityBytes, null, Gl.STREAM_DRAW);
                Gl.BufferSubData(Gl.ARRAY_BUFFER, 0, bytes, p);
            }
            Gl.DrawArraysInstanced(Gl.TRIANGLE_STRIP, 0, 4, _count);
            Gl.BindVertexArray(0);
            Gl.Disable(Gl.BLEND);
            _count = 0;
        }

        /// <summary>Releases GL objects.</summary>
        public void Dispose()
        {
            _program.Dispose();
            uint b = _quadVbo; Gl.DeleteBuffers(1, &b);
            b = _instVbo; Gl.DeleteBuffers(1, &b);
            uint v = _vao; Gl.DeleteVertexArrays(1, &v);
        }
    }
}
