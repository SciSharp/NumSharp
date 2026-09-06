using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace NumSharp.Interop.OpenBLAS
{
    /// <summary>
    ///     The dylib-relevant load commands of a Mach-O image: its install name, the libraries it
    ///     asks dyld to load, and its run-path entries.
    /// </summary>
    /// <remarks>
    ///     Read from the header only — the file is never mapped or read past
    ///     <c>sizeofcmds</c> — so inspecting the 25 MB OpenBLAS dylib costs one small read. Both
    ///     thin images (32- and 64-bit, either byte order) and fat/universal files are handled; a
    ///     fat file reports the UNION of every slice's dependencies, because the question asked
    ///     here is "what must exist next to this file", not "what will this CPU map".
    /// </remarks>
    internal sealed class MachOImage
    {
        /// <summary><c>LC_ID_DYLIB</c> — the name dyld records for this image, or null (executables, bundles).</summary>
        internal string InstallName { get; private set; }

        /// <summary>
        ///     Every <c>LC_LOAD_DYLIB</c> / <c>LC_LOAD_WEAK_DYLIB</c> / <c>LC_REEXPORT_DYLIB</c> /
        ///     <c>LC_LOAD_UPWARD_DYLIB</c> path, verbatim (<c>@loader_path/…</c>, <c>@rpath/…</c>,
        ///     absolute), in load-command order, de-duplicated.
        /// </summary>
        internal IReadOnlyList<string> Dependencies => _dependencies;

        /// <summary>Every <c>LC_RPATH</c> entry, verbatim.</summary>
        internal IReadOnlyList<string> RunPaths => _runPaths;

        private readonly List<string> _dependencies = new List<string>();
        private readonly List<string> _runPaths = new List<string>();

        private const uint MH_MAGIC = 0xFEEDFACE, MH_CIGAM = 0xCEFAEDFE;
        private const uint MH_MAGIC_64 = 0xFEEDFACF, MH_CIGAM_64 = 0xCFFAEDFE;
        private const uint FAT_MAGIC = 0xCAFEBABE, FAT_CIGAM = 0xBEBAFECA;
        private const uint FAT_MAGIC_64 = 0xCAFEBABF, FAT_CIGAM_64 = 0xBFBAFECA;

        private const uint LC_ID_DYLIB = 0xD;
        private const uint LC_LOAD_DYLIB = 0xC;
        private const uint LC_LOAD_WEAK_DYLIB = 0x80000018;
        private const uint LC_REEXPORT_DYLIB = 0x8000001F;
        private const uint LC_LOAD_UPWARD_DYLIB = 0x80000023;
        private const uint LC_RPATH = 0x8000001C;

        /// <summary>Upper bounds that keep a corrupt or hostile header from turning into a huge read.</summary>
        private const uint MaxCommands = 4096, MaxCommandBytes = 16 * 1024 * 1024, MaxFatSlices = 64;

        /// <summary>The <c>@loader_path/</c> prefix dyld expands to the directory of the image being loaded.</summary>
        internal const string LoaderPathPrefix = "@loader_path/";

        /// <summary>
        ///     Parses the load commands of the Mach-O file at <paramref name="path"/>.
        /// </summary>
        /// <returns>False when the file is not a Mach-O image (an ELF, a PE, garbage) or cannot be read.</returns>
        /// <remarks>Never throws — this runs on the module-initializer discovery path.</remarks>
        internal static bool TryRead(string path, out MachOImage image)
        {
            image = null;
            try
            {
                using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.None);
                return TryRead(stream, out image);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
            {
                return false;
            }
        }

        /// <summary>Stream form of <see cref="TryRead(string, out MachOImage)"/> (the stream must be seekable).</summary>
        internal static bool TryRead(Stream stream, out MachOImage image)
        {
            image = null;
            if (stream == null || !stream.CanSeek || stream.Length < 8)
                return false;

            Span<byte> magicBytes = stackalloc byte[8];
            stream.Position = 0;
            if (!ReadExactly(stream, magicBytes))
                return false;

            uint magicBe = BinaryPrimitives.ReadUInt32BigEndian(magicBytes);
            var result = new MachOImage();

            if (magicBe == FAT_MAGIC || magicBe == FAT_MAGIC_64)
            {
                // Fat headers are ALWAYS big-endian, whatever the slices are.
                bool fat64 = magicBe == FAT_MAGIC_64;
                uint nfat = BinaryPrimitives.ReadUInt32BigEndian(magicBytes.Slice(4));
                if (nfat == 0 || nfat > MaxFatSlices)
                    return false;

                int archSize = fat64 ? 32 : 20;
                var archs = new byte[archSize * nfat];
                stream.Position = 8;
                if (!ReadExactly(stream, archs))
                    return false;

                bool any = false;
                for (int i = 0; i < nfat; i++)
                {
                    var arch = new ReadOnlySpan<byte>(archs, i * archSize, archSize);
                    long offset = fat64
                        ? (long)BinaryPrimitives.ReadUInt64BigEndian(arch.Slice(8))
                        : BinaryPrimitives.ReadUInt32BigEndian(arch.Slice(8));
                    long size = fat64
                        ? (long)BinaryPrimitives.ReadUInt64BigEndian(arch.Slice(16))
                        : BinaryPrimitives.ReadUInt32BigEndian(arch.Slice(12));
                    if (offset < 0 || size < 28 || offset > stream.Length - size)
                        continue;
                    if (TryReadSlice(stream, offset, result))
                        any = true;
                }

                if (!any)
                    return false;
            }
            else
            {
                if (!TryReadSlice(stream, 0, result))
                    return false;
            }

            image = result;
            return true;
        }

        /// <summary>Parses one thin image at <paramref name="offset"/> into <paramref name="into"/>.</summary>
        private static bool TryReadSlice(Stream stream, long offset, MachOImage into)
        {
            Span<byte> header = stackalloc byte[32];
            stream.Position = offset;
            if (!ReadExactly(stream, header.Slice(0, 28)))
                return false;

            uint magicLe = BinaryPrimitives.ReadUInt32LittleEndian(header);
            bool littleEndian, is64;
            switch (magicLe)
            {
                case MH_MAGIC_64: littleEndian = true; is64 = true; break;
                case MH_CIGAM_64: littleEndian = false; is64 = true; break;
                case MH_MAGIC: littleEndian = true; is64 = false; break;
                case MH_CIGAM: littleEndian = false; is64 = false; break;
                default: return false;
            }

            // mach_header{,_64}: magic, cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags[, reserved]
            uint ncmds = ReadU32(header.Slice(16), littleEndian);
            uint sizeofcmds = ReadU32(header.Slice(20), littleEndian);
            int headerSize = is64 ? 32 : 28;
            if (ncmds == 0 || ncmds > MaxCommands || sizeofcmds < 8 || sizeofcmds > MaxCommandBytes)
                return false;
            if (offset + headerSize + sizeofcmds > stream.Length)
                return false;

            var commands = new byte[sizeofcmds];
            stream.Position = offset + headerSize;
            if (!ReadExactly(stream, commands))
                return false;

            int pos = 0;
            for (uint i = 0; i < ncmds; i++)
            {
                if (pos + 8 > commands.Length)
                    return false;

                var cmdSpan = new ReadOnlySpan<byte>(commands, pos, commands.Length - pos);
                uint cmd = ReadU32(cmdSpan, littleEndian);
                uint cmdsize = ReadU32(cmdSpan.Slice(4), littleEndian);
                if (cmdsize < 8 || cmdsize > cmdSpan.Length)
                    return false;

                switch (cmd)
                {
                    case LC_ID_DYLIB:
                        into.InstallName ??= ReadLcString(cmdSpan.Slice(0, (int)cmdsize), littleEndian);
                        break;
                    case LC_LOAD_DYLIB:
                    case LC_LOAD_WEAK_DYLIB:
                    case LC_REEXPORT_DYLIB:
                    case LC_LOAD_UPWARD_DYLIB:
                    {
                        var name = ReadLcString(cmdSpan.Slice(0, (int)cmdsize), littleEndian);
                        if (name != null && !into._dependencies.Contains(name))
                            into._dependencies.Add(name);
                        break;
                    }
                    case LC_RPATH:
                    {
                        var name = ReadLcString(cmdSpan.Slice(0, (int)cmdsize), littleEndian);
                        if (name != null && !into._runPaths.Contains(name))
                            into._runPaths.Add(name);
                        break;
                    }
                }

                pos += (int)cmdsize;
            }

            return true;
        }

        /// <summary>
        ///     The <c>lc_str</c> of a dylib/rpath command: a uint32 offset (from the command start)
        ///     at +8, then a NUL-terminated string within the command.
        /// </summary>
        private static string ReadLcString(ReadOnlySpan<byte> command, bool littleEndian)
        {
            if (command.Length < 12)
                return null;
            uint nameOffset = ReadU32(command.Slice(8), littleEndian);
            if (nameOffset < 12 || nameOffset >= command.Length)
                return null;
            var raw = command.Slice((int)nameOffset);
            int nul = raw.IndexOf((byte)0);
            if (nul >= 0)
                raw = raw.Slice(0, nul);
            return raw.Length == 0 ? null : Encoding.UTF8.GetString(raw);
        }

        private static uint ReadU32(ReadOnlySpan<byte> span, bool littleEndian)
            => littleEndian ? BinaryPrimitives.ReadUInt32LittleEndian(span) : BinaryPrimitives.ReadUInt32BigEndian(span);

        private static bool ReadExactly(Stream stream, Span<byte> buffer)
        {
            int total = 0;
            while (total < buffer.Length)
            {
                int n = stream.Read(buffer.Slice(total));
                if (n <= 0)
                    return false;
                total += n;
            }

            return true;
        }
    }
}
