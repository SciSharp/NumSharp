using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
 
namespace NumSharp
{
    public static partial class NDArrayExtensions
    {
        public static NDArray<TData> Array<TData>(this NDArray<TData> np, IEnumerable<TData> array, int ndim = 1)
        {
            np.Data = array.Select(x => x).ToArray();
            np.Shape[0] = np.Data.Length;

            return np;
        }
        
        public static NDArray<Byte> Array(this NDArray<Byte> np, System.Drawing.Bitmap image )
        {
            NDArray<Byte> imageArray = new NDArray<byte>();

            var bmpd = image.LockBits(new System.Drawing.Rectangle(0, 0, image.Width, image.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, image.PixelFormat);
            var dataSize = bmpd.Stride * bmpd.Height;

            imageArray.Data = new byte[dataSize];
            System.Runtime.InteropServices.Marshal.Copy(bmpd.Scan0, imageArray.Data, 0, imageArray.Data.Length);
            image.UnlockBits(bmpd);

            imageArray.Shape = new int[] { bmpd.Height, bmpd.Width, 3 };
    
            return imageArray;  
        }

        public static NDArray<TData[]> Array<TData>(this NDArray<TData[]> np, TData[][] array )
        {
            np.Data = array;

            return np;
        }

        public static NDArray<TData> Array<TData>(this NDArray<TData> np, TData[] array)
        {
            np.Data = array;
            np.Shape[0] = np.Data.Length;

            return np;
        }
    }
}
