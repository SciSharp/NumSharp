using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
 
namespace NumSharp
{
    public partial class NDArray<T>
    {
        public NDArray<T> array(IEnumerable<T> array, int ndim = 1)
        {
            this.Data = array.ToArray();
            this.Shape = new Shape(new int[] { this.Data.Length });

            return this;
        }
        
        public NDArray<Byte> array(System.Drawing.Bitmap image )
        {
            NDArray<Byte> imageArray = new NDArray<byte>();

            var bmpd = image.LockBits(new System.Drawing.Rectangle(0, 0, image.Width, image.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, image.PixelFormat);
            var dataSize = bmpd.Stride * bmpd.Height;

            imageArray.Data = new byte[dataSize];
            System.Runtime.InteropServices.Marshal.Copy(bmpd.Scan0, imageArray.Data, 0, imageArray.Data.Length);
            image.UnlockBits(bmpd);

            imageArray.Shape = new Shape(new int[] { bmpd.Height, bmpd.Width, 3 });
    
            return imageArray;  
        }

        public NDArray<T> array(T[][] array )
        {
            this.Data = new T[array.Length * array[0].Length];
            this.Shape = new Shape(array.Length,array[0].Length);

            for (int idx = 0; idx < array.Length;idx++)
                for (int jdx = 0; jdx < array[0].Length;jdx++)
                    this[idx,jdx] = array[idx][jdx];

            return this;
        }

    }
}
