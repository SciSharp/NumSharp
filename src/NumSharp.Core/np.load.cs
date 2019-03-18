using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace NumSharp.Core
{
  public static partial class np
  {
    #region Writer

    /// <summary>
    ///   Saves the specified array to an array of bytes.
    /// </summary>
    /// 
    /// <param name="array">The array to be saved to the array of bytes.</param>
    /// 
    /// <returns>A byte array containig the saved array.</returns>
    /// 
    public static byte[] Save(Array array)
    {
      using (var stream = new MemoryStream())
      {
        Save(array, stream);
        return stream.ToArray();
      }
    }

    /// <summary>
    ///   Saves the specified array to the disk using the npy format.
    /// </summary>
    /// 
    /// <param name="array">The array to be saved to disk.</param>
    /// <param name="path">The disk path under which the file will be saved.</param>
    /// 
    /// <returns>The number of bytes written when saving the file to disk.</returns>
    /// 
    public static ulong Save(Array array, string path)
    {
      using (var stream = new FileStream(path, FileMode.Create))
        return Save(array, stream);
    }

    /// <summary>
    ///   Saves the specified array to a stream using the npy format.
    /// </summary>
    /// 
    /// <param name="array">The array to be saved to disk.</param>
    /// <param name="stream">The stream to which the file will be saved.</param>
    /// 
    /// <returns>The number of bytes written when saving the file to disk.</returns>
    /// 
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

      return (ulong)buffer.LongLength;
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
        writtenBytes += (ulong)buffer.LongLength;
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

            writtenBytes += (ulong)buffer.LongLength;
          }
        }
      }

      return writtenBytes;
    }

    private static int writeHeader(BinaryWriter writer, string dtype, int[] shape)
    {
      // The first 6 bytes are a magic string: exactly "x93NUMPY"

      char[] magic = { 'N', 'U', 'M', 'P', 'Y' };
      writer.Write((byte)147);
      writer.Write(magic);
      writer.Write((byte)1); // major
      writer.Write((byte)0); // minor;

      string tuple = String.Join(", ", shape.Select(i => i.ToString()).ToArray());
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
      if (type == typeof(String))
        return "|S" + bytes;

      throw new NotSupportedException();
    }

    #endregion

    #region Reader

    /// <summary>
    ///   Loads an array of the specified type from a byte array.
    /// </summary>
    /// 
    /// <typeparam name="T">The type to be loaded from the npy-formatted file.</typeparam>
    /// <param name="bytes">The bytes that contain the matrix to be loaded.</param>
    /// 
    /// <returns>The array to be returned.</returns>
    /// 
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

    /// <summary>
    ///   Loads an array of the specified type from a file in the disk.
    /// </summary>
    /// 
    /// <typeparam name="T">The type to be loaded from the npy-formatted file.</typeparam>
    /// <param name="bytes">The bytes that contain the matrix to be loaded.</param>
    /// <param name="value">The object to be read. This parameter can be used to avoid the
    ///   need of specifying a generic argument to this function.</param>
    /// 
    /// <returns>The array to be returned.</returns>
    /// 
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

    /// <summary>
    ///   Loads an array of the specified type from a file in the disk.
    /// </summary>
    /// 
    /// <typeparam name="T">The type to be loaded from the npy-formatted file.</typeparam>
    /// <param name="path">The path to the file containing the matrix to be loaded.</param>
    /// <param name="value">The object to be read. This parameter can be used to avoid the
    ///   need of specifying a generic argument to this function.</param>
    /// 
    /// <returns>The array to be returned.</returns>
    /// 
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

    /// <summary>
    ///   Loads an array of the specified type from a stream.
    /// </summary>
    /// 
    /// <typeparam name="T">The type to be loaded from the npy-formatted file.</typeparam>
    /// <param name="stream">The stream containing the matrix to be loaded.</param>
    /// <param name="value">The object to be read. This parameter can be used to avoid the
    ///   need of specifying a generic argument to this function.</param>
    /// 
    /// <returns>The array to be returned.</returns>
    /// 
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

    /// <summary>
    ///   Loads an array of the specified type from a file in the disk.
    /// </summary>
    /// 
    /// <param name="path">The path to the file containing the matrix to be loaded.</param>
    /// 
    /// <returns>The array to be returned.</returns>
    /// 
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

    /// <summary>
    ///   Loads an array of the specified type from a stream.
    /// </summary>
    /// 
    /// <typeparam name="T">The type to be loaded from the npy-formatted file.</typeparam>
    /// <param name="stream">The stream containing the matrix to be loaded.</param>
    /// 
    /// <returns>The array to be returned.</returns>
    /// 
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

    /// <summary>
    ///   Loads a multi-dimensional array from an array of bytes.
    /// </summary>
    /// 
    /// <param name="bytes">The bytes that contain the matrix to be loaded.</param>
    /// 
    /// <returns>A multi-dimensional array containing the values available in the given stream.</returns>
    /// 
    public static Array LoadMatrix(byte[] bytes)
    {
      using (var stream = new MemoryStream(bytes))
        return LoadMatrix(stream);
    }

    /// <summary>
    ///   Loads a multi-dimensional array from a file in the disk.
    /// </summary>
    /// 
    /// <param name="path">The path to the file containing the matrix to be loaded.</param>
    /// 
    /// <returns>A multi-dimensional array containing the values available in the given stream.</returns>
    /// 
    public static Array LoadMatrix(string path)
    {
      using (var stream = new FileStream(path, FileMode.Open))
        return LoadMatrix(stream);
    }

    /// <summary>
    ///   Loads a jagged array from an array of bytes.
    /// </summary>
    /// 
    /// <param name="bytes">The bytes that contain the matrix to be loaded.</param>
    /// 
    /// <returns>A jagged array containing the values available in the given stream.</returns>
    /// 
    public static Array LoadJagged(byte[] bytes)
    {
      using (var stream = new MemoryStream(bytes))
        return LoadJagged(stream);
    }

    /// <summary>
    ///   Loads a jagged array from a file in the disk.
    /// </summary>
    /// 
    /// <param name="path">The path to the file containing the matrix to be loaded.</param>
    /// 
    /// <returns>A jagged array containing the values available in the given stream.</returns>
    /// 
    public static Array LoadJagged(string path)
    {
      using (var stream = new FileStream(path, FileMode.Open))
        return LoadJagged(stream);
    }

    /// <summary>
    ///   Loads a multi-dimensional array from a stream.
    /// </summary>
    /// 
    /// <param name="stream">The stream containing the matrix to be loaded.</param>
    /// 
    /// <returns>A multi-dimensional array containing the values available in the given stream.</returns>
    /// 
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

        Array matrix = Array.CreateInstance(type, shape);

        if (type == typeof(String))
          return readStringMatrix(reader, matrix, bytes, type, shape);
        return readValueMatrix(reader, matrix, bytes, type, shape);
      }
    }

    /// <summary>
    ///   Loads a jagged array from a stream.
    /// </summary>
    /// 
    /// <param name="stream">The stream containing the matrix to be loaded.</param>
    /// <param name="trim">Pass true to remove null or empty elements from the loaded array.</param>
    /// 
    /// <returns>A jagged array containing the values available in the given stream.</returns>
    /// 
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

        Array matrix = Array.CreateInstance(type, shape);

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
      shape = header.Substring(s, e - s).Split(',').Select(Int32.Parse).ToArray();

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

  }
}
