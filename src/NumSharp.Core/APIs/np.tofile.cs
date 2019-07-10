using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace NumSharp
{
    public partial class NDArray
    {
        // ndarray.tofile(fid, sep="", format="%s")
        public void tofile(string fileName)
        {
            switch (Type.GetTypeCode(dtype))
            {
                case TypeCode.Byte:
                {
                    using (BinaryWriter writer = new BinaryWriter(File.Open(fileName, FileMode.Create)))
                    {
                        writer.Write(Array as byte[]);
                    }
                }
                    break;
                case TypeCode.UInt16:
                {
                    var arr = Array as UInt16[];
                    using (BinaryWriter writer = new BinaryWriter(File.Open(fileName, FileMode.Create)))
                    {
                        for (var i = 0; i < arr.Length; i++)
                        {
                            writer.Write(arr[i]);
                        }
                    }
                }
                    break;
                default:
                    throw new NotImplementedException($"tofile dtype={dtype} not implemented yet");
            }
        }
    }
}
