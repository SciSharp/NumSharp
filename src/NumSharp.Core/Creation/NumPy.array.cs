using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
 
namespace NumSharp.Core
{
	public static partial class NumPyExtensions
	{
        public static NDArrayGeneric<T> array<T>(this NumPyGeneric<T> np, IEnumerable<T> array, int ndim = 1)
        {
            var nd = new NDArrayGeneric<T>();
            nd.Data = array.ToArray();
            nd.Shape = new Shape(new int[] { nd.Data.Length });

            return nd;
        }

        public static NDArray array<T>(this NumPy np, IEnumerable<T> array, int ndim = 1)
        {
			var nd = new NDArray(typeof(T));
			nd.Set(array.ToArray());
            nd.Shape = new Shape(new int[] { array.Count() });

            return nd;
        }

        public static NDArray array<T>(this NumPy np, System.Drawing.Bitmap image)
        {
            var imageArray = new NDArray(typeof(T));

            var bmpd = image.LockBits(new System.Drawing.Rectangle(0, 0, image.Width, image.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, image.PixelFormat);
            var dataSize = bmpd.Stride * bmpd.Height;

            imageArray.Set(new byte[dataSize]);
            System.Runtime.InteropServices.Marshal.Copy(bmpd.Scan0, imageArray.Storage.Bytes, 0, imageArray.Size);
            image.UnlockBits(bmpd);

            imageArray.Shape = new Shape(new int[] { bmpd.Height, bmpd.Width, System.Drawing.Image.GetPixelFormatSize(image.PixelFormat) / 8 });

            return imageArray;
        }

        public static NDArrayGeneric<Byte> array<T>(this NumPyGeneric<T> np, System.Drawing.Bitmap image )
        {
            NDArrayGeneric<Byte> imageArray = new NDArrayGeneric<Byte>();

            var bmpd = image.LockBits(new System.Drawing.Rectangle(0, 0, image.Width, image.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, image.PixelFormat);
            var dataSize = bmpd.Stride * bmpd.Height;

            imageArray.Data = new byte[dataSize];
            System.Runtime.InteropServices.Marshal.Copy(bmpd.Scan0, imageArray.Data, 0, imageArray.Data.Length);
            image.UnlockBits(bmpd);

            imageArray.Shape = new Shape(new int[] { bmpd.Height, bmpd.Width, System.Drawing.Image.GetPixelFormatSize(image.PixelFormat) / 8 });
    
            return imageArray;  
        }

        public static NDArray array<T>(this NumPy np, T[][] data)
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
            nd.Shape = new Shape(new int[] { data.Length, data[0].Length });

            return nd;
        }

        public static NDArrayGeneric<T> array<T>(this NumPyGeneric<T> np, T[][] data)
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

			var n = new NDArrayGeneric<T>();
			n.Data = all;
			n.Shape = new Shape(new int[] { data.Length, data[0].Length });

			return n;
		}

	}
}
