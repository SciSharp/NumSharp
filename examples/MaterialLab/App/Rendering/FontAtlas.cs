using System;
using System.Collections.Generic;
using NumSharp.Examples.MaterialLab.App.Native;

namespace NumSharp.Examples.MaterialLab.App.Rendering
{
    /// <summary>Where one glyph lives in the atlas and how far the pen advances after it.</summary>
    /// <param name="U0">Left texture coordinate.</param>
    /// <param name="V0">Top texture coordinate.</param>
    /// <param name="U1">Right texture coordinate.</param>
    /// <param name="V1">Bottom texture coordinate.</param>
    /// <param name="Width">Quad width in pixels (advance plus padding on both sides).</param>
    /// <param name="Height">Quad height in pixels.</param>
    /// <param name="Advance">Pen advance in pixels.</param>
    internal readonly record struct Glyph(float U0, float V0, float U1, float V1, float Width, float Height, float Advance);

    /// <summary>
    /// One typeface at one pixel size inside a <see cref="FontAtlas"/>: its glyph table and line metrics.
    /// </summary>
    internal sealed class FontFace
    {
        /// <summary>Glyph table keyed by character.</summary>
        public readonly Dictionary<char, Glyph> Glyphs = new Dictionary<char, Glyph>();
        /// <summary>Line height in pixels.</summary>
        public float LineHeight;
        /// <summary>Padding baked around every glyph cell (so anti-aliased edges and overhangs are not clipped).</summary>
        public float Pad;
        /// <summary>Human-readable face name.</summary>
        public string Name = "";

        /// <summary>
        /// Width of <paramref name="text"/> when drawn with this face (unknown characters advance like a space).
        /// </summary>
        /// <param name="text">Text to measure.</param>
        /// <returns>Width in pixels.</returns>
        public float Measure(ReadOnlySpan<char> text)
        {
            float w = 0;
            foreach (char c in text)
                w += Glyphs.TryGetValue(c, out var g) ? g.Advance : (Glyphs.TryGetValue(' ', out var sp) ? sp.Advance : 0);
            return w;
        }
    }

    /// <summary>
    /// A single texture holding several fonts rasterized by GDI (grayscale anti-aliased, white on black) plus
    /// a solid white block for flat fills. The HUD samples its red channel as coverage, so every overlay
    /// primitive — text in any face, panels, swatches, graph bars — shares one texture and draws in one
    /// instanced call.
    /// </summary>
    /// <remarks>
    /// GDI is used instead of a font library to keep the app dependency-free; the cost is paid once at
    /// startup (a few milliseconds). ClearType is deliberately avoided: its colored subpixel fringes would
    /// read as partial coverage in the other channels.
    /// </remarks>
    internal sealed unsafe class FontAtlas : IDisposable
    {
        /// <summary>Atlas texture size in pixels (all four faces fit with room to spare).</summary>
        private const int AtlasW = 1024, AtlasH = 512;

        /// <summary>The GL texture id.</summary>
        public uint Texture { get; private set; }
        /// <summary>UV of the center of the solid white block (for flat rectangles).</summary>
        public (float U, float V) White { get; private set; }
        /// <summary>Regular UI text.</summary>
        public FontFace Ui { get; private set; }
        /// <summary>Semibold UI text for headings.</summary>
        public FontFace UiBold { get; private set; }
        /// <summary>Monospaced digits for aligned numeric readouts.</summary>
        public FontFace Mono { get; private set; }
        /// <summary>The large display face for the FPS counter.</summary>
        public FontFace Big { get; private set; }

        /// <summary>Characters baked into every face beyond printable ASCII: the few symbols the HUD uses.</summary>
        private const string Extra = "·×²³µ→←↑↓↔≈±…—°▸●■◆";

        /// <summary>
        /// Rasterizes the faces into a DIB and uploads it. Requires a current GL context.
        /// </summary>
        /// <param name="uiScale">Size multiplier for high-DPI or fullscreen layouts (1 = base sizes).</param>
        /// <exception cref="InvalidOperationException">GDI could not create the DIB section.</exception>
        public FontAtlas(float uiScale = 1f)
        {
            nint dc = Win32.CreateCompatibleDC(0);
            var bmi = new Win32.BITMAPINFOHEADER
            {
                biSize = (uint)sizeof(Win32.BITMAPINFOHEADER),
                biWidth = AtlasW,
                biHeight = -AtlasH,   // top-down rows: row 0 is the top, matching our V=0-at-top UVs
                biPlanes = 1,
                biBitCount = 32,
            };
            void* bits;
            nint bmp = Win32.CreateDIBSection(dc, &bmi, Win32.DIB_RGB_COLORS, &bits, 0, 0);
            if (bmp == 0 || bits == null)
                throw new InvalidOperationException("CreateDIBSection failed for the font atlas.");
            nint oldBmp = Win32.SelectObject(dc, bmp);
            new Span<byte>(bits, AtlasW * AtlasH * 4).Clear();
            Win32.SetBkMode(dc, Win32.TRANSPARENT);
            Win32.SetTextColor(dc, 0x00FFFFFF);

            int penX = 2, penY = 2, shelfH = 0;
            // Solid white block for flat fills (8x8, sampled at its center so bilinear never reaches black).
            var px = (uint*)bits;
            for (int y = 0; y < 8; y++) for (int x = 0; x < 8; x++) px[(penY + y) * AtlasW + penX + x] = 0xFFFFFFFF;
            White = ((penX + 4f) / AtlasW, (penY + 4f) / AtlasH);
            penX += 12; shelfH = 10;

            Ui = Bake(dc, "Segoe UI", (int)Math.Round(15 * uiScale), Win32.FW_NORMAL, ref penX, ref penY, ref shelfH);
            UiBold = Bake(dc, "Segoe UI Semibold", (int)Math.Round(15 * uiScale), Win32.FW_SEMIBOLD, ref penX, ref penY, ref shelfH);
            Mono = Bake(dc, "Consolas", (int)Math.Round(14 * uiScale), Win32.FW_NORMAL, ref penX, ref penY, ref shelfH);
            Big = Bake(dc, "Segoe UI Light", (int)Math.Round(40 * uiScale), 300, ref penX, ref penY, ref shelfH);

            Win32.GdiFlush();
            uint tex;
            Gl.GenTextures(1, &tex);
            Texture = tex;
            Gl.BindTexture(Gl.TEXTURE_2D, tex);
            Gl.PixelStorei(Gl.UNPACK_ALIGNMENT, 4);
            Gl.TexImage2D(Gl.TEXTURE_2D, 0, Gl.RGBA8, AtlasW, AtlasH, 0, Gl.BGRA, Gl.UNSIGNED_BYTE, bits);
            Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_MIN_FILTER, Gl.LINEAR);
            Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_MAG_FILTER, Gl.LINEAR);
            Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_WRAP_S, Gl.CLAMP_TO_EDGE);
            Gl.TexParameteri(Gl.TEXTURE_2D, Gl.TEXTURE_WRAP_T, Gl.CLAMP_TO_EDGE);

            Win32.SelectObject(dc, oldBmp);
            Win32.DeleteObject(bmp);
            Win32.DeleteDC(dc);
        }

        /// <summary>
        /// Rasterizes one face into the DIB with simple shelf packing, recording each glyph's cell.
        /// </summary>
        /// <param name="dc">Memory DC with the DIB selected.</param>
        /// <param name="faceName">Typeface (GDI substitutes a fallback when missing).</param>
        /// <param name="pixelHeight">Em height in pixels.</param>
        /// <param name="weight">Font weight.</param>
        /// <param name="penX">Packing cursor x (advanced).</param>
        /// <param name="penY">Packing cursor y (advanced).</param>
        /// <param name="shelfH">Current shelf height (updated).</param>
        /// <returns>The baked face.</returns>
        /// <exception cref="InvalidOperationException">The atlas ran out of room (a UI scale too large for 1024×512).</exception>
        private static FontFace Bake(nint dc, string faceName, int pixelHeight, int weight, ref int penX, ref int penY, ref int shelfH)
        {
            nint font = Win32.CreateFontW(-pixelHeight, 0, 0, 0, weight, 0, 0, 0, Win32.DEFAULT_CHARSET, Win32.OUT_TT_PRECIS,
                Win32.CLIP_DEFAULT_PRECIS, Win32.ANTIALIASED_QUALITY, Win32.DEFAULT_PITCH, faceName);
            nint oldFont = Win32.SelectObject(dc, font);
            const int pad = 2;
            var face = new FontFace { Pad = pad, Name = faceName };

            var chars = new List<char>();
            for (char c = ' '; c <= '~'; c++) chars.Add(c);
            chars.AddRange(Extra);
            foreach (char c in chars)
            {
                Win32.SIZE sz;
                char ch = c;
                Win32.GetTextExtentPoint32W(dc, &ch, 1, &sz);
                int cw = sz.cx + pad * 2, chH = sz.cy + pad * 2;
                face.LineHeight = Math.Max(face.LineHeight, sz.cy);
                if (penX + cw >= AtlasW) { penX = 2; penY += shelfH + 2; shelfH = 0; }
                if (penY + chH >= AtlasH) throw new InvalidOperationException("Font atlas is full — lower the UI scale.");
                Win32.TextOutW(dc, penX + pad, penY + pad, &ch, 1);
                face.Glyphs[c] = new Glyph((float)penX / AtlasW, (float)penY / AtlasH, (float)(penX + cw) / AtlasW, (float)(penY + chH) / AtlasH,
                    cw, chH, sz.cx);
                penX += cw + 2;
                shelfH = Math.Max(shelfH, chH);
            }
            // Start the next face on a fresh shelf so faces never interleave.
            penX = 2; penY += shelfH + 4; shelfH = 0;
            Win32.SelectObject(dc, oldFont);
            Win32.DeleteObject(font);
            return face;
        }

        /// <summary>Deletes the texture.</summary>
        public void Dispose()
        {
            if (Texture != 0)
            {
                uint t = Texture;
                Gl.DeleteTextures(1, &t);
                Texture = 0;
            }
        }
    }
}
