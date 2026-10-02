using System;
using NumSharp.Examples.MaterialLab.App.Native;

namespace NumSharp.Examples.MaterialLab.App.Rendering
{
    /// <summary>
    /// An offscreen framebuffer with one or more color textures of a single format — the renderer's unit of
    /// intermediate storage (particle splat fields, blurred fields, the lit scene, bloom, the final image).
    /// Multiple attachments are written in ONE pass via <c>glDrawBuffers</c> (multiple render targets),
    /// which is how a single particle draw accumulates both a material's color and its per-category
    /// weights.
    /// </summary>
    /// <remarks>
    /// Resizing reallocates the textures (their contents are lost), so the renderer rebuilds targets only
    /// when the window size actually changes.
    /// </remarks>
    internal sealed unsafe class RenderTarget : IDisposable
    {
        /// <summary>The framebuffer object.</summary>
        private uint _fbo;
        /// <summary>One texture per color attachment.</summary>
        private readonly uint[] _tex;
        /// <summary>Internal format of every attachment (e.g. RGBA16F).</summary>
        private readonly int _internalFormat;
        /// <summary>Pixel format and component type used when (re)allocating the textures.</summary>
        private readonly uint _format, _type;
        /// <summary>Bilinear (true) or nearest (false) sampling.</summary>
        private readonly bool _linear;

        /// <summary>Width in pixels.</summary>
        public int Width { get; private set; }
        /// <summary>Height in pixels.</summary>
        public int Height { get; private set; }
        /// <summary>The framebuffer object id.</summary>
        public uint Fbo => _fbo;

        /// <summary>
        /// Creates a target with <paramref name="attachments"/> color textures.
        /// </summary>
        /// <param name="width">Width in pixels (≥ 1).</param>
        /// <param name="height">Height in pixels (≥ 1).</param>
        /// <param name="attachments">Number of color textures (1–4).</param>
        /// <param name="internalFormat">GL internal format, e.g. <see cref="Gl.RGBA16F"/> for HDR accumulation.</param>
        /// <param name="format">Client format paired with the internal format (e.g. <see cref="Gl.RGBA"/>).</param>
        /// <param name="type">Client component type (e.g. <see cref="Gl.FLOAT"/>).</param>
        /// <param name="linear">Bilinear filtering (true for fields sampled at other resolutions) vs nearest.</param>
        /// <exception cref="InvalidOperationException">The driver reports the framebuffer incomplete (unsupported format).</exception>
        public RenderTarget(int width, int height, int attachments, int internalFormat, uint format, uint type, bool linear = true)
        {
            _tex = new uint[attachments];
            _internalFormat = internalFormat;
            _format = format;
            _type = type;
            _linear = linear;
            Allocate(width, height);
        }

        /// <summary>The texture id of attachment <paramref name="i"/>.</summary>
        /// <param name="i">Attachment index.</param>
        /// <returns>The GL texture name.</returns>
        public uint Texture(int i) => _tex[i];

        /// <summary>Reallocates the textures if the size changed.</summary>
        /// <param name="width">New width.</param>
        /// <param name="height">New height.</param>
        /// <returns>True when reallocation happened (contents are undefined afterwards).</returns>
        public bool Resize(int width, int height)
        {
            width = Math.Max(1, width);
            height = Math.Max(1, height);
            if (width == Width && height == Height) return false;
            Release();
            Allocate(width, height);
            return true;
        }

        /// <summary>
        /// Binds this target for drawing, sets the viewport to cover it and enables every attachment as a
        /// render target.
        /// </summary>
        public void Bind()
        {
            Gl.BindFramebuffer(Gl.FRAMEBUFFER, _fbo);
            Gl.Viewport(0, 0, Width, Height);
            uint* bufs = stackalloc uint[4];
            for (int i = 0; i < _tex.Length; i++) bufs[i] = Gl.COLOR_ATTACHMENT0 + (uint)i;
            Gl.DrawBuffers(_tex.Length, bufs);
        }

        /// <summary>Binds attachment <paramref name="attachment"/> to texture unit <paramref name="unit"/> for sampling.</summary>
        /// <param name="attachment">Attachment index.</param>
        /// <param name="unit">Texture unit (0-based).</param>
        public void BindTexture(int attachment, int unit)
        {
            Gl.ActiveTexture(Gl.TEXTURE0 + (uint)unit);
            Gl.BindTexture(Gl.TEXTURE_2D, _tex[attachment]);
        }

        /// <summary>Creates the textures and the FBO.</summary>
        /// <param name="width">Width.</param>
        /// <param name="height">Height.</param>
        /// <exception cref="InvalidOperationException">Framebuffer incomplete.</exception>
        private void Allocate(int width, int height)
        {
            Width = Math.Max(1, width);
            Height = Math.Max(1, height);
            fixed (uint* t = _tex) Gl.GenTextures(_tex.Length, t);
            for (int i = 0; i < _tex.Length; i++)
            {
                Gl.BindTexture(Gl.TEXTURE_2D, _tex[i]);
                Gl.TexImage2D(Gl.TEXTURE_2D, 0, _internalFormat, Width, Height, 0, _format, _type, null);
                int filter = _linear ? Gl.LINEAR : Gl.NEAREST;
                Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_MIN_FILTER, filter);
                Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_MAG_FILTER, filter);
                Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_WRAP_S, Gl.CLAMP_TO_EDGE);
                Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_WRAP_T, Gl.CLAMP_TO_EDGE);
            }
            uint fbo;
            Gl.GenFramebuffers(1, &fbo);
            _fbo = fbo;
            Gl.BindFramebuffer(Gl.FRAMEBUFFER, _fbo);
            for (int i = 0; i < _tex.Length; i++)
                Gl.FramebufferTexture2D(Gl.FRAMEBUFFER, Gl.COLOR_ATTACHMENT0 + (uint)i, Gl.TEXTURE_2D, _tex[i], 0);
            uint status = Gl.CheckFramebufferStatus(Gl.FRAMEBUFFER);
            Gl.BindFramebuffer(Gl.FRAMEBUFFER, 0);
            if (status != Gl.FRAMEBUFFER_COMPLETE)
                throw new InvalidOperationException($"Framebuffer incomplete (status 0x{status:X}) for format 0x{_internalFormat:X} {Width}x{Height}.");
        }

        /// <summary>Deletes the FBO and textures.</summary>
        private void Release()
        {
            if (_fbo != 0)
            {
                uint f = _fbo;
                Gl.DeleteFramebuffers(1, &f);
                _fbo = 0;
            }
            fixed (uint* t = _tex) Gl.DeleteTextures(_tex.Length, t);
            Array.Clear(_tex);
        }

        /// <summary>Releases GPU memory.</summary>
        public void Dispose() => Release();
    }
}
