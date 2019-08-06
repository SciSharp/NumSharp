using System.IO;

namespace NumSharp
{
    public partial class NDArray
    {

        /// <summary>
        ///     Write array to a file as text or binary (default).<br></br>
        ///     Data is always written in ‘C’ order, independent of the order of a. <br></br>The data produced by this method can be recovered using the function fromfile().
        /// </summary>
        /// <param name="fid">An open file object, or a string containing a filename.</param>
        /// <remarks>https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.tofile.html</remarks>
        public void tofile(string fid)
        {
            //TODO! support for sliced data (if sliced, clone and then save)
            unsafe
            {
                using (var fs = File.Open(fid, FileMode.OpenOrCreate))
                using (var ums = new UnmanagedMemoryStream((byte*)this.Array.Address, this.Array.BytesLength))
                {
                    fs.SetLength(0);
                    ums.CopyTo(fs);
                }
            }
        }
    }
}
