using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NumSharp.Utilities;

namespace NumSharp
{
    public static partial class np
    {
        #region NpyFormat

        //Signature from numpy doc: 
        //   numpy.load(file, mmap_mode=None, allow_pickle=True, fix_imports=True, encoding='ASCII')[source]
        public static NDArray load(string path)
        {
            using (var stream = new FileStream(path, FileMode.Open))
                return load(stream);
        }

        public static NDArray load(Stream stream)
        {
            using (var reader = new BinaryReader(stream, System.Text.Encoding.ASCII
#if !NET35 && !NET40
                , leaveOpen: true
#endif
            ))
            {
                int bytes;
                Type type;
                int[] shape;
                if (!parseReader(reader, out bytes, out type, out shape))
                    throw new FormatException();

                Array array = Arrays.Create(type, shape.Aggregate((dims, dim) => dims * dim));

                var result = new NDArray(readValueMatrix(reader, array, bytes, type, shape));
                return result.reshape(shape);
            }
        }

        public static T Load<T>(byte[] bytes)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable
#if !NET35
            , IStructuralComparable, IStructuralEquatable
#endif
        {
            if (typeof(T).IsArray && (typeof(T).GetElementType().IsArray || typeof(T).GetElementType() == typeof(string)))
                return LoadJagged(bytes) as T;
            return LoadMatrix(bytes) as T;
        }

        public static T Load<T>(byte[] bytes, out T value)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable
#if !NET35
            , IStructuralComparable, IStructuralEquatable
#endif
        {
            return value = Load<T>(bytes);
        }


        public static T Load<T>(string path, out T value)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable
#if !NET35
            , IStructuralComparable, IStructuralEquatable
#endif
        {
            return value = Load<T>(path);
        }

        public static T Load<T>(Stream stream, out T value)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable
#if !NET35
            , IStructuralComparable, IStructuralEquatable
#endif
        {
            return value = Load<T>(stream);
        }


        public static T Load<T>(string path)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable
#if !NET35
            , IStructuralComparable, IStructuralEquatable
#endif
        {
            using (var stream = new FileStream(path, FileMode.Open))
                return Load<T>(stream);
        }


        public static T Load<T>(Stream stream)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable
#if !NET35
            , IStructuralComparable, IStructuralEquatable
#endif
        {
            if (typeof(T).IsArray && (typeof(T).GetElementType().IsArray || typeof(T).GetElementType() == typeof(string)))
                return LoadJagged(stream) as T;
            return LoadMatrix(stream) as T;
        }

        public static Array LoadMatrix(byte[] bytes)
        {
            using (var stream = new MemoryStream(bytes))
                return LoadMatrix(stream);
        }


        public static Array LoadMatrix(string path)
        {
            using (var stream = new FileStream(path, FileMode.Open))
                return LoadMatrix(stream);
        }


        public static Array LoadJagged(byte[] bytes)
        {
            using (var stream = new MemoryStream(bytes))
                return LoadJagged(stream);
        }

        public static Array LoadJagged(string path)
        {
            using (var stream = new FileStream(path, FileMode.Open))
                return LoadJagged(stream);
        }

        public static Array LoadMatrix(Stream stream)
        {
            using (var reader = new BinaryReader(stream, System.Text.Encoding.ASCII
#if !NET35 && !NET40
                , leaveOpen: true
#endif
            ))
            {
                int bytes;
                Type type;
                int[] shape;
                if (!parseReader(reader, out bytes, out type, out shape))
                    throw new FormatException();

                Array matrix = Arrays.Create(type, shape);

                if (type == typeof(String))
                    return readStringMatrix(reader, matrix, bytes, type, shape);
                return readValueMatrix(reader, matrix, bytes, type, shape);
            }
        }


        public static Array LoadJagged(Stream stream, bool trim = true)
        {
            using (var reader = new BinaryReader(stream, System.Text.Encoding.ASCII
#if !NET35 && !NET40
                , leaveOpen: true
#endif
            ))
            {
                int bytes;
                Type type;
                int[] shape;
                if (!parseReader(reader, out bytes, out type, out shape))
                    throw new FormatException();

                Array matrix = Arrays.Create(type, shape);

                if (type == typeof(String))
                {
                    Array result = readStringMatrix(reader, matrix, bytes, type, shape);

                    //if (trim)
                    //    return result.Trim();
                    return result;
                }

                return readValueJagged(reader, matrix, bytes, type, shape);
            }
        }

        private static Array readValueMatrix(BinaryReader reader, Array matrix, int bytes, Type type, int[] shape)
        {
            int total = 1;
            for (int i = 0; i < shape.Length; i++)
                total *= shape[i];
            var buffer = new byte[bytes * total];

            reader.Read(buffer, 0, buffer.Length);
            Buffer.BlockCopy(buffer, 0, matrix, 0, buffer.Length);

            return matrix;
        }

        static IEnumerable<int[]> GetIndices(Array array, int[] current, int pos)
        {
            if (pos == current.Length)
                yield return current;
            else
            {
                for (int i = 0; i < array.GetLength(pos); i++)
                {
                    current[pos] = i;
                    foreach (var ind in GetIndices(array, current, pos + 1))
                        yield return ind;
                    current[pos] = 0;
                }
            }
        }

        static IEnumerable<int[]> GetIndices(Array array)
        {
            return GetIndices(array, new int[array.Rank], 0);
        }

        private static Array readValueJagged(BinaryReader reader, Array matrix, int bytes, Type type, int[] shape)
        {
            int last = shape[shape.Length - 1];
            byte[] buffer = new byte[bytes * last];

            int[] firsts = new int[shape.Length - 1];
            for (int i = 0; i < firsts.Length; i++)
                firsts[i] = -1;

            foreach (var p in GetIndices(matrix))
            {
                bool changed = false;
                for (int i = 0; i < firsts.Length; i++)
                {
                    if (firsts[i] != p[i])
                    {
                        firsts[i] = p[i];
                        changed = true;
                    }
                }

                if (!changed)
                    continue;

                Array arr = (Array)matrix.GetValue(indices: firsts);

                reader.Read(buffer, 0, buffer.Length);
                Buffer.BlockCopy(buffer, 0, arr, 0, buffer.Length);
            }

            return matrix;
        }

        private static Array readStringMatrix(BinaryReader reader, Array matrix, int bytes, Type type, int[] shape)
        {
            var buffer = new byte[bytes];

            unsafe
            {
                fixed (byte* b = buffer)
                {
                    foreach (var p in GetIndices(matrix))
                    {
                        reader.Read(buffer, 0, bytes);
                        if (buffer[0] == byte.MinValue)
                        {
                            bool isNull = true;
                            for (int i = 1; i < buffer.Length; i++)
                            {
                                if (buffer[i] != byte.MaxValue)
                                {
                                    isNull = false;
                                    break;
                                }
                            }

                            if (isNull)
                            {
                                matrix.SetValue(value: null, indices: p);
                                continue;
                            }
                        }

#if NETSTANDARD1_4
                        String s = new String((char*)b);
#else
                        String s = new String((sbyte*)b);
#endif
                        matrix.SetValue(value: s, indices: p);
                    }
                }
            }

            return matrix;
        }

        private static bool parseReader(BinaryReader reader, out int bytes, out Type t, out int[] shape)
        {
            bytes = 0;
            t = null;
            shape = null;

            // The first 6 bytes are a magic string: exactly "x93NUMPY"
            if (reader.ReadChar() != 63) return false;
            if (reader.ReadChar() != 'N') return false;
            if (reader.ReadChar() != 'U') return false;
            if (reader.ReadChar() != 'M') return false;
            if (reader.ReadChar() != 'P') return false;
            if (reader.ReadChar() != 'Y') return false;

            byte major = reader.ReadByte(); // 1
            byte minor = reader.ReadByte(); // 0

            if (major != 1 || minor != 0)
                throw new NotSupportedException();

            ushort len = reader.ReadUInt16();

            string header = new String(reader.ReadChars(len));
            string mark = "'descr': '";
            int s = header.IndexOf(mark) + mark.Length;
            int e = header.IndexOf("'", s + 1);
            string type = header.Substring(s, e - s);
            bool? isLittleEndian;
            t = GetType(type, out bytes, out isLittleEndian);

            if (isLittleEndian.HasValue && isLittleEndian.Value == false)
                throw new Exception();

            mark = "'fortran_order': ";
            s = header.IndexOf(mark) + mark.Length;
            e = header.IndexOf(",", s + 1);
            bool fortran = bool.Parse(header.Substring(s, e - s));

            if (fortran)
                throw new Exception();

            mark = "'shape': (";
            s = header.IndexOf(mark) + mark.Length;
            e = header.IndexOf(")", s + 1);
            shape = header.Substring(s, e - s).Split(',').Where(v => !String.IsNullOrEmpty(v)).Select(Int32.Parse).ToArray();

            return true;
        }

        private static Type GetType(string dtype, out int bytes, out bool? isLittleEndian)
        {
            isLittleEndian = IsLittleEndian(dtype);
            bytes = Int32.Parse(dtype.Substring(2));

            string typeCode = dtype.Substring(1);

            if (typeCode == "b1")
                return typeof(bool);
            if (typeCode == "i1")
                return typeof(Byte);
            if (typeCode == "i2")
                return typeof(Int16);
            if (typeCode == "i4")
                return typeof(Int32);
            if (typeCode == "i8")
                return typeof(Int64);
            if (typeCode == "u1")
                return typeof(Byte);
            if (typeCode == "u2")
                return typeof(UInt16);
            if (typeCode == "u4")
                return typeof(UInt32);
            if (typeCode == "u8")
                return typeof(UInt64);
            if (typeCode == "f4")
                return typeof(Single);
            if (typeCode == "f8")
                return typeof(Double);
            if (typeCode.StartsWith("S"))
                return typeof(String);

            throw new NotSupportedException();
        }

        private static bool? IsLittleEndian(string type)
        {
            bool? littleEndian = null;

            switch (type[0])
            {
                case '<':
                    littleEndian = true;
                    break;
                case '>':
                    littleEndian = false;
                    break;
                case '|':
                    littleEndian = null;
                    break;
                default:
                    throw new Exception();
            }

            return littleEndian;
        }

        #endregion

        #region NpzFormat

        public static void Load_Npz<T>(byte[] bytes, out T value)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            using (var dict = Load_Npz<T>(bytes))
            {
                value = dict.Values.First();
            }
        }

        public static void Load_Npz<T>(string path, out T value)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            using (var dict = Load_Npz<T>(path))
            {
                value = dict.Values.First();
            }
        }

        public static void Load_Npz<T>(Stream stream, out T value)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            using (var dict = Load_Npz<T>(stream))
            {
                value = dict.Values.First();
            }
        }

        public static NpzDictionary<T> Load_Npz<T>(byte[] bytes)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return Load_Npz<T>(new MemoryStream(bytes));
        }

        public static NpzDictionary<T> Load_Npz<T>(string path, out NpzDictionary<T> value)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return value = Load_Npz<T>(new FileStream(path, FileMode.Open));
        }

        public static NpzDictionary<T> Load_Npz<T>(Stream stream, out NpzDictionary<T> value)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return value = Load_Npz<T>(stream);
        }

        public static NpzDictionary<T> Load_Npz<T>(string path)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return Load_Npz<T>(new FileStream(path, FileMode.Open));
        }

        public static NpzDictionary<T> Load_Npz<T>(Stream stream)
            where T : class,
#if !NETSTANDARD1_4
            ICloneable,
#endif
            IList, ICollection, IEnumerable, IStructuralComparable, IStructuralEquatable
        {
            return new NpzDictionary<T>(stream);
        }

        public static NpzDictionary<Array> LoadMatrix_Npz(byte[] bytes)
        {
            return LoadMatrix_Npz(new MemoryStream(bytes));
        }

        public static NpzDictionary<Array> LoadMatrix_Npz(string path)
        {
            return LoadMatrix_Npz(new FileStream(path, FileMode.Open));
        }

        public static NpzDictionary<Array> LoadMatrix_Npz(Stream stream)
        {
            return new NpzDictionary(stream, jagged: false);
        }

        public static NpzDictionary<Array> LoadJagged_Npz(byte[] bytes)
        {
            return LoadJagged_Npz(new MemoryStream(bytes));
        }

        public static NpzDictionary<Array> LoadJagged_Npz(string path)
        {
            return LoadJagged_Npz(new FileStream(path, FileMode.Open));
        }

        public static NpzDictionary<Array> LoadJagged_Npz(Stream stream, bool trim = true)
        {
            return new NpzDictionary(stream, jagged: true);
        }

        #endregion
    }
}
