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
                        var nd = new NDArray(dtype);
                        using (BinaryReader reader = new BinaryReader(File.Open(fileName, FileMode.Open)))
                        {
                            var dataArray = reader.ReadBytes((int)arraySize);
                            nd.SetData(dataArray);
                        }
                        return nd;
                    }
            }
            throw new NotImplementedException($"fromfile dtype={dtype} not implemented yet");
        }

    }
}
