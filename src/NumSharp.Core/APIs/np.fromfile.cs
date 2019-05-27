using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace NumSharp
{
    public static partial class np
    {
        // NumPy Signature: numpy.fromfile(file, dtype=float, count=-1, sep='')
        public static NDArray fromfile(string fileName, Type dtype)
        {
            switch (Type.GetTypeCode(dtype))
            {
                case TypeCode.Byte:
                    {
                        var fileSize = new System.IO.FileInfo(fileName).Length;
                        var arraySize = fileSize / Marshal.SizeOf(dtype);
                        byte[] dataArray;
                        using (BinaryReader reader = new BinaryReader(File.Open(fileName, FileMode.Open)))
                        {
                            dataArray = reader.ReadBytes((int)arraySize);
                        }
                        return new NDArray(dataArray);
                    }
                case TypeCode.UInt16:
                    {
                        var fileSize = new System.IO.FileInfo(fileName).Length;
                        var arraySize = fileSize / Marshal.SizeOf(dtype);
                        var dataArray = new ushort[arraySize];
                        using (BinaryReader reader = new BinaryReader(File.Open(fileName, FileMode.Open)))
                        {
                            for (var i = 0; i < (int)arraySize; i++)
                            {
                                dataArray[i] = reader.ReadUInt16();
                            }
                        }
                        return new NDArray(dataArray);
                    }
            }
            throw new NotImplementedException($"fromfile dtype={dtype} not implemented yet");
        }

    }
}
