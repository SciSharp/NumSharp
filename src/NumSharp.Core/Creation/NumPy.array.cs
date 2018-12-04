using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
 
namespace NumSharp.Core
{
	public partial class NumPy
	{
        public NDArray array(Array array, Type dtype = null, int ndim = 1)
        {
            dtype = (dtype == null) ? array.GetType().GetElementType() : dtype;
            
			var nd = new NDArray(dtype);

            if ((array.Rank == 1) && ( !array.GetType().GetElementType().IsArray ))
			{
                nd.Storage = NDStorage.CreateByShapeAndType	(dtype, new Shape(new int[] { array.Length }));
                nd.Storage.SetData(array); 
            }
            else 
            {
                throw new Exception("Method is not implemeneted for multidimensional arrays or jagged arrays.");
            }
            
            return nd;
        }

        public NDArray array(System.Drawing.Bitmap image)
        {
            var imageArray = new NDArray(typeof(Byte));

            var bmpd = image.LockBits(new System.Drawing.Rectangle(0, 0, image.Width, image.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, image.PixelFormat);
            var dataSize = bmpd.Stride * bmpd.Height;

            var bytes = new byte[dataSize];
            System.Runtime.InteropServices.Marshal.Copy(bmpd.Scan0, bytes, 0, dataSize);
            image.UnlockBits(bmpd);

            imageArray.Set(bytes);
            imageArray.Storage.Shape = new Shape(new int[] { bmpd.Height, bmpd.Width, System.Drawing.Image.GetPixelFormatSize(image.PixelFormat) / 8 });

            return imageArray;
        }

        public NDArray array<T>(T[][] data)
        {
            int size = data.Length * data[0].Length;
            var all = new T[size];

            int idx = 0;
            for (int row = 0; row < data.Length; row++)
            {
                for (int col = 0; col < data[row].Length; col++)
                {
                    all[idx] = data[row][col];
                    idx++;
                }
            }

            var nd = new NDArray(typeof(T));
            nd.Set(all.ToArray());
            nd.Storage.Shape = new Shape(new int[] { data.Length, data[0].Length });

            return nd;
        }
	}
}
