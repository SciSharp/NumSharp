using System;
using System.IO;
using System.IO.Compression;
using System.Text;

namespace NumSharp.Examples.MaterialLab.App.Rendering
{
    /// <summary>
    /// A minimal, dependency-free PNG encoder (8-bit RGB, one IDAT, zlib via the BCL) — carried over from the
    /// falling-sand Player so screenshots need no image library.
    /// </summary>
    internal static class PngWriter
    {
        /// <summary>CRC-32 lookup table (built on first use).</summary>
        private static uint[] _crcTable;

        /// <summary>Writes top-to-bottom RGB pixels as a PNG file.</summary>
        /// <param name="path">Output file path (directories must exist).</param>
        /// <param name="rgb">Pixels, <c>width*height*3</c> bytes, row 0 = top.</param>
        /// <param name="width">Width in pixels.</param>
        /// <param name="height">Height in pixels.</param>
        /// <exception cref="ArgumentException"><paramref name="rgb"/> is smaller than width×height×3.</exception>
        /// <exception cref="IOException">The file cannot be written.</exception>
        public static void Write(string path, byte[] rgb, int width, int height)
        {
            if (rgb.Length < width * height * 3) throw new ArgumentException("pixel buffer too small", nameof(rgb));
            using var fs = File.Create(path);
            fs.Write(new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 });   // PNG signature
            var ihdr = new byte[13];
            WriteBE(ihdr, 0, width); WriteBE(ihdr, 4, height);
            ihdr[8] = 8;    // bit depth
            ihdr[9] = 2;    // colour type 2 = truecolour RGB
            WriteChunk(fs, "IHDR", ihdr);
            // Each scanline: filter byte 0 (none) + the row.
            var raw = new byte[height * (1 + width * 3)];
            int p = 0;
            for (int y = 0; y < height; y++)
            {
                raw[p++] = 0;
                Array.Copy(rgb, y * width * 3, raw, p, width * 3);
                p += width * 3;
            }
            byte[] compressed;
            using (var ms = new MemoryStream())
            {
                using (var z = new ZLibStream(ms, CompressionLevel.Optimal, leaveOpen: true)) z.Write(raw, 0, raw.Length);
                compressed = ms.ToArray();
            }
            WriteChunk(fs, "IDAT", compressed);
            WriteChunk(fs, "IEND", Array.Empty<byte>());
        }

        /// <summary>Writes one chunk: length, type, data, CRC-32 over type+data.</summary>
        private static void WriteChunk(Stream s, string type, byte[] data)
        {
            var len = new byte[4]; WriteBE(len, 0, data.Length); s.Write(len);
            var t = Encoding.ASCII.GetBytes(type); s.Write(t);
            s.Write(data);
            var c = new byte[4]; WriteBE(c, 0, (int)Crc32(t, data)); s.Write(c);
        }

        /// <summary>Writes a big-endian 32-bit integer.</summary>
        private static void WriteBE(byte[] b, int o, int v)
        {
            b[o] = (byte)(v >> 24); b[o + 1] = (byte)(v >> 16); b[o + 2] = (byte)(v >> 8); b[o + 3] = (byte)v;
        }

        /// <summary>Standard PNG CRC-32 (polynomial 0xEDB88320) over the chunk type then its data.</summary>
        private static uint Crc32(byte[] type, byte[] data)
        {
            if (_crcTable == null)
            {
                var table = new uint[256];
                for (uint n = 0; n < 256; n++)
                {
                    uint c = n;
                    for (int k = 0; k < 8; k++) c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
                    table[n] = c;
                }
                _crcTable = table;
            }
            uint crc = 0xFFFFFFFFu;
            foreach (byte b in type) crc = _crcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
            foreach (byte b in data) crc = _crcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
            return crc ^ 0xFFFFFFFFu;
        }
    }
}
