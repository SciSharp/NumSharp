using System;
using NumSharp.Examples.MaterialLab.App.Native;
using NumSharp.Examples.MaterialLab.Simulation;

namespace NumSharp.Examples.MaterialLab.App.Rendering
{
    /// <summary>What the renderer should draw for the brush cursor.</summary>
    /// <param name="X">Center x in simulation units.</param>
    /// <param name="Y">Center y in simulation units.</param>
    /// <param name="Radius">Radius in simulation units.</param>
    /// <param name="Visible">Whether the ring is drawn.</param>
    /// <param name="Color">Ring color (linear RGB).</param>
    internal readonly record struct BrushCursor(float X, float Y, float Radius, bool Visible, (float R, float G, float B) Color);

    /// <summary>
    /// Turns the simulation state into a lit HDR image, then a tone-mapped 8-bit frame. Owns every GPU
    /// resource of the pipeline described in <see cref="Shaders"/>: the particle instance stream, the obstacle
    /// distance-field texture, the half-resolution field targets, the grain layer, the HDR scene, the bloom
    /// chain and the final image.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>CPU cost.</b> Per frame the CPU packs 8 floats per particle into a staging array and uploads it
    /// once (orphaning the buffer so the driver never stalls on the previous frame), re-uploads the obstacle
    /// field only when <see cref="MpmGrid.ObstacleVersion"/> changes, and issues about fifteen draw calls. All
    /// the shading work is on the GPU.
    /// </para>
    /// <para>
    /// <b>Final image.</b> The finished frame lives in <see cref="Final"/> (an offscreen RGBA8 target), which
    /// the HUD draws into and the window then shows — so a screenshot reads exactly what the user sees,
    /// even from a hidden window.
    /// </para>
    /// </remarks>
    internal sealed unsafe class SceneRenderer : IDisposable
    {
        /// <summary>Instance stride: x, y, vx, vy, material, color seed, temperature, J.</summary>
        private const int FloatsPerParticle = 8;

        /// <summary>Shader programs: field splat, grain sprites, field blur, composite, bloom threshold, bloom blur, tone map, blit.</summary>
        private readonly GlProgram _field, _grain, _blur, _composite, _bright, _blur1, _tone, _copy;
        /// <summary>
        /// Render targets: the half-resolution field pair (splat → blur), the full-resolution grain layer, the HDR
        /// scene, the quarter-resolution bloom pair, and the final 8-bit image the HUD draws into.
        /// </summary>
        private RenderTarget _fieldA, _fieldB, _grains, _scene, _bloomA, _bloomB, _final;
        /// <summary>Particle vertex array, unit quad, instance buffer, the attribute-less VAO for full-screen passes, and the wall texture.</summary>
        private uint _particleVao, _quadVbo, _instVbo, _emptyVao, _phiTex;
        /// <summary>Obstacle version and size of the uploaded wall texture (re-uploaded only when the walls change).</summary>
        private int _phiVersion = -1, _phiW, _phiH;
        /// <summary>CPU staging of the particle instances (grows on demand).</summary>
        private float[] _staging = new float[FloatsPerParticle * 65536];
        /// <summary>Current size of the particle instance buffer on the GPU.</summary>
        private int _instCapacityBytes;
        /// <summary>Per-material linear albedo (uniform array, index = material).</summary>
        private readonly float[] _albedo = new float[16 * 3];
        /// <summary>Per-material render category: 0 liquid, 1 grain, 2 solid.</summary>
        private readonly float[] _category = new float[16];
        /// <summary>Per-material translucency, gloss, emission and color jitter.</summary>
        private readonly float[] _props = new float[16 * 4];
        /// <summary>Per-material grain sprite size relative to the particle spacing (snow flakes are larger).</summary>
        private readonly float[] _grainScale = new float[16];
        /// <summary>Per-material foaminess (how white its spray turns).</summary>
        private readonly float[] _foam = new float[16];

        /// <summary>Output width in pixels.</summary>
        public int Width { get; private set; }
        /// <summary>Output height in pixels.</summary>
        public int Height { get; private set; }
        /// <summary>The finished 8-bit frame (HUD is drawn into it afterwards).</summary>
        public RenderTarget Final => _final;
        /// <summary>Bloom intensity (0 disables the glow).</summary>
        public float BloomStrength { get; set; } = 0.55f;
        /// <summary>Exposure before tone mapping.</summary>
        public float Exposure { get; set; } = 1.15f;

        /// <summary>Compiles the programs and allocates the targets. Requires a current GL context.</summary>
        /// <param name="width">Output width.</param>
        /// <param name="height">Output height.</param>
        /// <exception cref="InvalidOperationException">A shader failed to compile/link (message has the log).</exception>
        public SceneRenderer(int width, int height)
        {
            _field = new GlProgram("field", Shaders.ParticleVs, Shaders.FieldFs, "aCorner", "aPosVel", "aMatSeed");
            _grain = new GlProgram("grain", Shaders.ParticleVs, Shaders.GrainFs, "aCorner", "aPosVel", "aMatSeed");
            _blur = new GlProgram("blur", Shaders.FullscreenVs, Shaders.BlurFs);
            _composite = new GlProgram("composite", Shaders.FullscreenVs, Shaders.CompositeFs);
            _bright = new GlProgram("bright", Shaders.FullscreenVs, Shaders.BrightFs);
            _blur1 = new GlProgram("blur1", Shaders.FullscreenVs, Shaders.Blur1Fs);
            _tone = new GlProgram("tone", Shaders.FullscreenVs, Shaders.ToneFs);
            _copy = new GlProgram("copy", Shaders.FullscreenVs, Shaders.CopyFs);

            uint vao, qvbo, ivbo, evao, phi;
            Gl.GenVertexArrays(1, &vao); Gl.GenBuffers(1, &qvbo); Gl.GenBuffers(1, &ivbo); Gl.GenVertexArrays(1, &evao);
            _particleVao = vao; _quadVbo = qvbo; _instVbo = ivbo; _emptyVao = evao;
            Gl.BindVertexArray(_particleVao);
            float* corners = stackalloc float[] { -1, -1, 1, -1, -1, 1, 1, 1 };
            Gl.BindBuffer(Gl.ARRAY_BUFFER, _quadVbo);
            Gl.BufferData(Gl.ARRAY_BUFFER, 8 * sizeof(float), corners, Gl.STATIC_DRAW);
            Gl.EnableVertexAttribArray(0);
            Gl.VertexAttribPointer(0, 2, Gl.FLOAT, 0, 2 * sizeof(float), 0);
            Gl.BindBuffer(Gl.ARRAY_BUFFER, _instVbo);
            int stride = FloatsPerParticle * sizeof(float);
            Gl.EnableVertexAttribArray(1);
            Gl.VertexAttribPointer(1, 4, Gl.FLOAT, 0, stride, 0);
            Gl.VertexAttribDivisor(1, 1);
            Gl.EnableVertexAttribArray(2);
            Gl.VertexAttribPointer(2, 4, Gl.FLOAT, 0, stride, 4 * sizeof(float));
            Gl.VertexAttribDivisor(2, 1);
            Gl.BindVertexArray(0);

            Gl.GenTextures(1, &phi);
            _phiTex = phi;

            for (int i = 0; i < Materials.Count; i++)
            {
                var m = Materials.All[i];
                _albedo[i * 3] = m.Color.R; _albedo[i * 3 + 1] = m.Color.G; _albedo[i * 3 + 2] = m.Color.B;
                _category[i] = (float)m.Category;
                _props[i * 4] = m.Translucency; _props[i * 4 + 1] = m.Gloss; _props[i * 4 + 2] = m.Emission; _props[i * 4 + 3] = m.ColorJitter;
                // Sand grains slightly overlap their neighbours; snow flakes overlap a lot and feather out.
                _grainScale[i] = m.Id == MaterialId.Snow ? 1.55f : 1.08f;
                _foam[i] = m.Foaminess;
            }
            Resize(width, height);
        }

        /// <summary>(Re)allocates every target for a new output size (no-op when unchanged).</summary>
        /// <param name="width">Output width.</param>
        /// <param name="height">Output height.</param>
        public void Resize(int width, int height)
        {
            width = Math.Max(16, width); height = Math.Max(16, height);
            if (width == Width && height == Height) return;
            Width = width; Height = height;
            DisposeTargets();
            int hw = Math.Max(8, width / 2), hh = Math.Max(8, height / 2);
            _fieldA = new RenderTarget(hw, hh, 3, Gl.RGBA16F, Gl.RGBA, Gl.FLOAT);
            _fieldB = new RenderTarget(hw, hh, 3, Gl.RGBA16F, Gl.RGBA, Gl.FLOAT);
            _grains = new RenderTarget(width, height, 1, Gl.RGBA16F, Gl.RGBA, Gl.FLOAT);
            _scene = new RenderTarget(width, height, 1, Gl.RGBA16F, Gl.RGBA, Gl.FLOAT);
            int qw = Math.Max(8, width / 4), qh = Math.Max(8, height / 4);
            _bloomA = new RenderTarget(qw, qh, 1, Gl.RGBA16F, Gl.RGBA, Gl.FLOAT);
            _bloomB = new RenderTarget(qw, qh, 1, Gl.RGBA16F, Gl.RGBA, Gl.FLOAT);
            _final = new RenderTarget(width, height, 1, Gl.RGBA8, Gl.RGBA, Gl.UNSIGNED_BYTE);
        }

        /// <summary>
        /// Renders the world into <see cref="Final"/>.
        /// </summary>
        /// <param name="world">The simulation.</param>
        /// <param name="brush">Brush cursor to draw.</param>
        /// <param name="time">Seconds since start (animates lava crust and similar effects).</param>
        public void Render(MpmWorld world, in BrushCursor brush, float time)
        {
            int n = PackParticles(world);
            UploadObstacles(world.Grid);
            float domainW = world.Grid.Width, domainH = 1f;
            float spacing = 0.5f * world.Grid.Dx;
            var light = Normalize(-0.45f, 0.62f, 0.64f);

            // 1. Field splat (half res, additive, three targets).
            _fieldA.Bind();
            Gl.ClearColor(0, 0, 0, 0);
            Gl.Clear(Gl.COLOR_BUFFER_BIT);
            Gl.Enable(Gl.BLEND);
            Gl.BlendEquation(Gl.FUNC_ADD);
            Gl.BlendFunc(Gl.ONE, Gl.ONE);
            _field.Use();
            SetParticleUniforms(_field, domainW, domainH, spacing, mode: 0);
            DrawParticles(n);

            // 2. Blur the field (horizontal into B, vertical back into A). Taps are 1.6 texels apart: a
            //    σ ≈ 3 half-res px (≈ 6 screen px) kernel smooths particle noise out of the surface normals while
            //    keeping edges within one particle spacing.
            Gl.Disable(Gl.BLEND);
            BlurField(_fieldA, _fieldB, 1.6f / _fieldA.Width, 0f);
            BlurField(_fieldB, _fieldA, 0f, 1.6f / _fieldA.Height);

            // 3. Grains (full res, premultiplied "over").
            _grains.Bind();
            Gl.ClearColor(0, 0, 0, 0);
            Gl.Clear(Gl.COLOR_BUFFER_BIT);
            Gl.Enable(Gl.BLEND);
            Gl.BlendFunc(Gl.ONE, Gl.ONE_MINUS_SRC_ALPHA);
            _grain.Use();
            SetParticleUniforms(_grain, domainW, domainH, spacing, mode: 1);
            _grain.Set("uLight", light.X, light.Y, light.Z);
            DrawParticles(n);
            Gl.Disable(Gl.BLEND);

            // 4. Composite into the HDR scene.
            _scene.Bind();
            _composite.Use();
            _fieldA.BindTexture(0, 0); _composite.Set("uField0", 0);
            _fieldA.BindTexture(1, 1); _composite.Set("uField1", 1);
            _fieldA.BindTexture(2, 2); _composite.Set("uField2", 2);
            _grains.BindTexture(0, 3); _composite.Set("uGrains", 3);
            Gl.ActiveTexture(Gl.TEXTURE0 + 4); Gl.BindTexture(Gl.TEXTURE_2D, _phiTex); _composite.Set("uPhi", 4);
            _composite.Set("uResolution", Width, Height);
            _composite.Set("uFieldTexel", 1f / _fieldA.Width, 1f / _fieldA.Height);
            _composite.Set("uDomain", domainW, domainH);
            _composite.Set("uGridSize", world.Grid.Nx, world.Grid.Ny);
            _composite.Set("uCellPx", Height / (float)world.Grid.Ny);
            _composite.Set("uIso", 2.2f);
            _composite.Set("uTime", time);
            _composite.Set("uLight", light.X, light.Y, light.Z);
            _composite.Set("uBrush", brush.X, brush.Y, brush.Radius, brush.Visible ? 1f : 0f);
            _composite.Set("uBrushColor", brush.Color.R, brush.Color.G, brush.Color.B);
            FullscreenPass();

            // 5. Bloom: bright-pass at quarter res, two blur iterations.
            _bloomA.Bind();
            _bright.Use();
            _scene.BindTexture(0, 0); _bright.Set("uTex", 0); _bright.Set("uThreshold", 1.0f);
            FullscreenPass();
            for (int it = 0; it < 2; it++)
            {
                Blur1(_bloomA, _bloomB, 1f / _bloomA.Width, 0f);
                Blur1(_bloomB, _bloomA, 0f, 1f / _bloomA.Height);
            }

            // 6. Tone map into the final 8-bit image.
            _final.Bind();
            _tone.Use();
            _scene.BindTexture(0, 0); _tone.Set("uScene", 0);
            _bloomA.BindTexture(0, 1); _tone.Set("uBloom", 1);
            _tone.Set("uBloomStrength", BloomStrength);
            _tone.Set("uExposure", Exposure);
            FullscreenPass();
        }

        /// <summary>Copies <see cref="Final"/> to the window's back buffer.</summary>
        /// <param name="windowWidth">Window client width.</param>
        /// <param name="windowHeight">Window client height.</param>
        public void Present(int windowWidth, int windowHeight)
        {
            Gl.BindFramebuffer(Gl.FRAMEBUFFER, 0);
            Gl.Viewport(0, 0, windowWidth, windowHeight);
            _copy.Use();
            _final.BindTexture(0, 0);
            _copy.Set("uTex", 0);
            FullscreenPass();
        }

        /// <summary>Reads <see cref="Final"/> back as top-down RGB bytes (screenshots).</summary>
        /// <returns>width·height·3 bytes, row 0 = top.</returns>
        public byte[] ReadFinalRgb()
        {
            Gl.BindFramebuffer(Gl.FRAMEBUFFER, _final.Fbo);
            var rgba = new byte[Width * Height * 4];
            Gl.PixelStorei(Gl.PACK_ALIGNMENT, 1);
            fixed (byte* p = rgba) Gl.ReadPixels(0, 0, Width, Height, Gl.RGBA, Gl.UNSIGNED_BYTE, p);
            var rgb = new byte[Width * Height * 3];
            for (int y = 0; y < Height; y++)
            {
                int src = (Height - 1 - y) * Width * 4, dst = y * Width * 3;   // GL rows run bottom-up
                for (int x = 0; x < Width; x++)
                {
                    rgb[dst + x * 3] = rgba[src + x * 4];
                    rgb[dst + x * 3 + 1] = rgba[src + x * 4 + 1];
                    rgb[dst + x * 3 + 2] = rgba[src + x * 4 + 2];
                }
            }
            return rgb;
        }

        /// <summary>Packs every particle as (x, y, vx, vy, material, seed, temperature, J) into the staging array.</summary>
        /// <param name="world">The simulation.</param>
        /// <returns>Particle count.</returns>
        private int PackParticles(MpmWorld world)
        {
            int total = world.ParticleCount;
            if (_staging.Length < total * FloatsPerParticle) Array.Resize(ref _staging, Math.Max(total * FloatsPerParticle, _staging.Length * 2));
            int o = 0;
            fixed (float* dst = _staging)
            {
                foreach (var g in world.Groups)
                {
                    int count = g.Count;
                    if (count == 0) continue;
                    float mat = (float)g.Material.Id;
                    float* X = g.Ptr(PField.X), Y = g.Ptr(PField.Y), VX = g.Ptr(PField.VX), VY = g.Ptr(PField.VY);
                    float* S = g.Ptr(PField.Seed), T = g.Ptr(PField.Temp), J = g.Ptr(PField.Jp);
                    for (int p = 0; p < count; p++)
                    {
                        float* d = dst + o;
                        d[0] = X[p]; d[1] = Y[p]; d[2] = VX[p]; d[3] = VY[p];
                        d[4] = mat; d[5] = S[p]; d[6] = T[p]; d[7] = J[p];
                        o += FloatsPerParticle;
                    }
                }
                int bytes = o * sizeof(float);
                Gl.BindBuffer(Gl.ARRAY_BUFFER, _instVbo);
                if (bytes > _instCapacityBytes) _instCapacityBytes = Math.Max(bytes, _instCapacityBytes * 2);
                Gl.BufferData(Gl.ARRAY_BUFFER, Math.Max(_instCapacityBytes, 64), null, Gl.STREAM_DRAW);
                if (bytes > 0) Gl.BufferSubData(Gl.ARRAY_BUFFER, 0, bytes, dst);
            }
            return o / FloatsPerParticle;
        }

        /// <summary>Re-uploads the (4× resolution) obstacle distance field when the grid reports a change.</summary>
        /// <param name="grid">The grid.</param>
        private void UploadObstacles(MpmGrid grid)
        {
            if (grid.ObstacleVersion == _phiVersion && grid.Nx == _phiW && grid.Ny == _phiH) return;
            Gl.BindTexture(Gl.TEXTURE_2D, _phiTex);
            Gl.PixelStorei(Gl.UNPACK_ALIGNMENT, 4);
            // R32F: 0x822E. Row 0 is the floor, which is GL's bottom row: no flip. The fine field covers exactly the
            // grid's extent, so the composite's (x·Ny + ½)/(Nx, Ny) lookup addresses it unchanged.
            fixed (float* fine = grid.PhiFine)
                Gl.TexImage2D(Gl.TEXTURE_2D, 0, 0x822E, grid.FineW, grid.FineH, 0, Gl.RED, Gl.FLOAT, fine);
            Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_MIN_FILTER, Gl.LINEAR);
            Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_MAG_FILTER, Gl.LINEAR);
            Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_WRAP_S, Gl.CLAMP_TO_EDGE);
            Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_WRAP_T, Gl.CLAMP_TO_EDGE);
            _phiVersion = grid.ObstacleVersion; _phiW = grid.Nx; _phiH = grid.Ny;
        }

        /// <summary>Sets the uniforms shared by the particle programs.</summary>
        private void SetParticleUniforms(GlProgram p, float domainW, float domainH, float spacing, int mode)
        {
            p.Set("uDomain", domainW, domainH);
            p.Set("uRadiusField", spacing * 2.4f);
            p.Set("uRadiusGrain", spacing);
            p.Set("uMode", mode);
            fixed (float* c = _category) Gl.Uniform1fv(p.U("uCategory"), 16, c);
            fixed (float* g = _grainScale) Gl.Uniform1fv(p.U("uGrainScale"), 16, g);
            fixed (float* f = _foam) Gl.Uniform1fv(p.U("uFoam"), 16, f);
            p.SetVec3Array("uAlbedo", _albedo);
            p.SetVec4Array("uProps", _props);
        }

        /// <summary>Draws every particle as an instanced quad.</summary>
        private void DrawParticles(int n)
        {
            if (n == 0) return;
            Gl.BindVertexArray(_particleVao);
            Gl.DrawArraysInstanced(Gl.TRIANGLE_STRIP, 0, 4, n);
            Gl.BindVertexArray(0);
        }

        /// <summary>One separable blur pass over the three field targets.</summary>
        private void BlurField(RenderTarget src, RenderTarget dst, float dx, float dy)
        {
            dst.Bind();
            _blur.Use();
            src.BindTexture(0, 0); _blur.Set("uT0", 0);
            src.BindTexture(1, 1); _blur.Set("uT1", 1);
            src.BindTexture(2, 2); _blur.Set("uT2", 2);
            _blur.Set("uDir", dx, dy);
            FullscreenPass();
        }

        /// <summary>One separable blur pass over a single target.</summary>
        private void Blur1(RenderTarget src, RenderTarget dst, float dx, float dy)
        {
            dst.Bind();
            _blur1.Use();
            src.BindTexture(0, 0); _blur1.Set("uTex", 0);
            _blur1.Set("uDir", dx, dy);
            FullscreenPass();
        }

        /// <summary>Draws the full-screen triangle with the current program.</summary>
        private void FullscreenPass()
        {
            Gl.BindVertexArray(_emptyVao);
            Gl.DrawArrays(Gl.TRIANGLES, 0, 3);
            Gl.BindVertexArray(0);
        }

        /// <summary>Normalizes a 3-vector.</summary>
        private static (float X, float Y, float Z) Normalize(float x, float y, float z)
        {
            float l = MathF.Sqrt(x * x + y * y + z * z);
            return (x / l, y / l, z / l);
        }

        /// <summary>Releases the render targets.</summary>
        private void DisposeTargets()
        {
            _fieldA?.Dispose(); _fieldB?.Dispose(); _grains?.Dispose(); _scene?.Dispose();
            _bloomA?.Dispose(); _bloomB?.Dispose(); _final?.Dispose();
        }

        /// <summary>Releases every GPU resource.</summary>
        public void Dispose()
        {
            DisposeTargets();
            _field.Dispose(); _grain.Dispose(); _blur.Dispose(); _composite.Dispose();
            _bright.Dispose(); _blur1.Dispose(); _tone.Dispose(); _copy.Dispose();
            uint b = _quadVbo; Gl.DeleteBuffers(1, &b);
            b = _instVbo; Gl.DeleteBuffers(1, &b);
            uint v = _particleVao; Gl.DeleteVertexArrays(1, &v);
            v = _emptyVao; Gl.DeleteVertexArrays(1, &v);
            uint t = _phiTex; Gl.DeleteTextures(1, &t);
        }
    }
}
