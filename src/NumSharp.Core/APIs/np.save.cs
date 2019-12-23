using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;

namespace NumSharp
{
    public static partial class np
    {
        #region NpyFormat

        public static void save(string filepath, Array arr)
        {
            Save(arr, filepath);
        }

        public static void save(string filepath, NDArray arr)
        {
            Save((Array)arr, filepath);
        }

        public static byte[] Save(Array array)
        {
            using (var stream = new MemoryStream())
            {
                Save(array, stream);
                return stream.ToArray();
            }
        }


        public static ulong Save(Array array, string path)
        {
            if (Path.GetExtension(path) != ".npy")
            {
                path += ".npy";
            }

            using (var stream = new FileStream(path, FileMode.Create))
                return Save(array, stream);
        }

        public static ulong Save(Array array, Stream stream)
        {
            using (var writer = new BinaryWriter(stream
#if !NET35 && !NET40
                , System.Text.Encoding.ASCII, leaveOpen: true
#endif
            ))
            {
                Type type;
                int maxLength;
                string dtype = GetDtypeFromType(array, out type, out maxLength);

                int[] shape = Enumerable.Range(0, array.Rank).Select(d => array.GetLength(d)).ToArray();

                ulong bytesWritten = (ulong)writeHeader(writer, dtype, shape);

                if (array.GetType().GetElementType().IsArray || array.GetType().GetElementType() == typeof(string))
                {
                    if (type == typeof(String))
                        return bytesWritten + writeStringMatrix(writer, array, maxLength, shape);
                    return bytesWritten + writeValueJagged(writer, array, maxLength, shape);
                }
                else
                {
                    if (type == typeof(String))
                        return bytesWritten + writeStringMatrix(writer, array, maxLength, shape);
                    return bytesWritten + writeValueMatrix(writer, array, maxLength, shape);
                }
            }
        }

        private static ulong writeValueMatrix(BinaryWriter reader, Array matrix, int bytes, int[] shape)
        {
            int total = 1;
            for (int i = 0; i < shape.Length; i++)
                total *= shape[i];
            var buffer = new byte[bytes * total];

            Buffer.BlockCopy(matrix, 0, buffer, 0, buffer.Length);
            reader.Write(buffer, 0, buffer.Length);

#if NETSTANDARD1_4
            return (ulong)buffer.Length;
#else
            return (ulong)buffer.LongLength;
#endif
        }

        static IEnumerable<T> Enumerate<T>(Array a, int[] dimensions, int pos)
        {
            if (pos == dimensions.Length - 1)
            {
                for (int i = 0; i < dimensions[pos]; i++)
                    yield return (T)a.GetValue(i);
            }
            else
            {
                for (int i = 0; i < dimensions[pos]; i++)
                    foreach (var subArray in Enumerate<T>(a.GetValue(i) as Array, dimensions, pos + 1))
                        yield return subArray;
            }
        }

        private static ulong writeValueJagged(BinaryWriter reader, Array matrix, int bytes, int[] shape)
        {
            int last = shape[shape.Length - 1];
            byte[] buffer = new byte[bytes * last];
            int[] first = shape.Take(shape.Length - 1).ToArray();

            ulong writtenBytes = 0;
            foreach (Array arr in Enumerate<Array>(matrix, first, 0))
            {
                Array.Clear(buffer, arr.Length, buffer.Length - buffer.Length);
                Buffer.BlockCopy(arr, 0, buffer, 0, buffer.Length);
                reader.Write(buffer, 0, buffer.Length);
#if NETSTANDARD1_4
                writtenBytes += (ulong)buffer.Length;
#else
                writtenBytes += (ulong)buffer.LongLength;
#endif
            }

            return writtenBytes;
        }

        private static ulong writeStringMatrix(BinaryWriter reader, Array matrix, int bytes, int[] shape)
        {
            var buffer = new byte[bytes];
            var empty = new byte[bytes];
            empty[0] = byte.MinValue;
            for (int i = 1; i < empty.Length; i++)
                empty[i] = byte.MaxValue;

            ulong writtenBytes = 0;

            unsafe
            {
                fixed (byte* b = buffer)
                {
                    foreach (String s in Enumerate<String>(matrix, shape, 0))
                    {
                        if (s != null)
                        {
                            int c = 0;
                            for (int i = 0; i < s.Length; i++)
                                b[c++] = (byte)s[i];
                            for (; c < buffer.Length; c++)
                                b[c] = byte.MinValue;

                            reader.Write(buffer, 0, bytes);
                        }
                        else
                        {
                            reader.Write(empty, 0, bytes);
                        }

#if NETSTANDARD1_4
                        writtenBytes += (ulong)buffer.Length;
#else
                        writtenBytes += (ulong)buffer.LongLength;
#endif
                    }
                }
            }

            return writtenBytes;
        }

        private static int writeHeader(BinaryWriter writer, string dtype, int[] shape)
        {
            // The first 6 bytes are a magic string: exactly "x93NUMPY"

            char[] magic = {'N', 'U', 'M', 'P', 'Y'};
            writer.Write((byte)147);
            writer.Write(magic);
            writer.Write((byte)1); // major
            writer.Write((byte)0); // minor;

            string tuple = String.Join(", ", shape.Select(i => i.ToString()).ToArray());
            if (shape.Length == 1)
                tuple += ","; // 1-dim array's shape is (R,)
            string header = "{{'descr': '{0}', 'fortran_order': False, 'shape': ({1}), }}";
            header = String.Format(header, dtype, tuple);
            int preamble = 10; // magic string (6) + 4

            int len = header.Length + 1; // the 1 is to account for the missing \n at the end
            int headerSize = len + preamble;

            int pad = 16 - (headerSize % 16);
            header = header.PadRight(header.Length + pad);
            header += "\n";
            headerSize = header.Length + preamble;

            if (headerSize % 16 != 0)
                throw new Exception();

            writer.Write((ushort)header.Length);
            for (int i = 0; i < header.Length; i++)
                writer.Write((byte)header[i]);

            return headerSize;
        }

        static Type GetInnerMostType(Type arrayType)
        {
            if (arrayType.GetElementType().IsArray)
                return GetInnerMostType(arrayType.GetElementType());
            return arrayType.GetElementType();
        }

        private static string GetDtypeFromType(Array array, out Type type, out int bytes)
        {
            type = GetInnerMostType(array.GetType());

            bytes = 1;

            if (type == typeof(String))
            {
                int[] shape = Enumerable.Range(0, array.Rank).Select(d => array.GetLength(d)).ToArray();
                foreach (String s in Enumerate<String>(array, shape, 0))
                {
                    if (s.Length > bytes)
                        bytes = s.Length;
                }
            }
            else if (type == typeof(bool))
            {
                bytes = 1;
            }
            else
            {
#pragma warning disable 618 // SizeOf would be Obsolete
                bytes = Marshal.SizeOf(type);
#pragma warning restore 618 // SizeOf would be Obsolete
            }

            if (type == typeof(bool))
                return "|b1";
            if (type == typeof(Byte))
                return "|i1";
            if (type == typeof(Int16))
                return "<i2";
            if (type == typeof(Int32))
                return "<i4";
            if (type == typeof(Int64))
                return "<i8";
            if (type == typeof(Single))
                return "<f4";
            if (type == typeof(Double))
                return "<f8";
            if (type == typeof(UInt16))
                return "<u2";
            if (type == typeof(UInt32))
                return "<u4";
            if (type == typeof(UInt64))
                return "<u8";
            if (type == typeof(String))
                return "|S" + bytes;

            throw new NotSupportedException();
        }

        #endregion

        #region NpzFormat

        const CompressionLevel DEFAULT_COMPRESSION = CompressionLevel.Fastest;

        public static byte[] Save_Npz(Dictionary<string, Array> arrays, CompressionLevel compression = DEFAULT_COMPRESSION)
        {
            using (var stream = new MemoryStream())
            {
                Save_Npz(arrays, stream, compression, leaveOpen: true);
                return stream.ToArray();
            }
        }

        public static byte[] Save_Npz(Array array, CompressionLevel compression = DEFAULT_COMPRESSION)
        {
            using (var stream = new MemoryStream())
            {
                Save_Npz(array, stream, compression, leaveOpen: true);
                return stream.ToArray();
            }
        }


        public static void Save_Npz(Dictionary<string, Array> arrays, string path, CompressionLevel compression = DEFAULT_COMPRESSION)
        {
            using (var stream = new FileStream(path, FileMode.Create))
            {
                Save_Npz(arrays, stream, compression);
            }
        }

        public static void Save_Npz(Array array, string path, CompressionLevel compression = DEFAULT_COMPRESSION)
        {
            using (var stream = new FileStream(path, FileMode.Create))
            {
                Save_Npz(array, stream, compression);
            }
        }

        public static void Save_Npz(Dictionary<string, Array> arrays, Stream stream, CompressionLevel compression = DEFAULT_COMPRESSION, bool leaveOpen = false)
        {
            using (var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: leaveOpen))
            {
                foreach (KeyValuePair<string, Array> p in arrays)
                {
                    var entry = zip.CreateEntry(p.Key, compression);
                    using (Stream s = entry.Open())
                    {
                        Save(p.Value, s);
                    }
                }
            }
        }

        public static void Save_Npz(Array array, Stream stream, CompressionLevel compression = DEFAULT_COMPRESSION, bool leaveOpen = false)
        {
            using (var zip = new ZipArchive(stream, ZipArchiveMode.Create, leaveOpen: leaveOpen))
            {
                var entry = zip.CreateEntry("arr_0");
                using (Stream s = entry.Open())
                {
                    Save(array, s);
                }
                
            }
        }

        #endregion
    }
}
