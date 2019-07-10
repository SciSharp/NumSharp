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
            unsafe
            {
                using (var fs = File.Open(fileName, FileMode.OpenOrCreate))
                using (var ums = new UnmanagedMemoryStream((byte*)this.Array.Address, this.Array.Count * this.dtypesize))
                {
                    fs.SetLength(0);
                    ums.CopyTo(fs);
                }
            }
        }
    }
}
