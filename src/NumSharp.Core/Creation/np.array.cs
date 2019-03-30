using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Drawing;
using NumSharp.Backends.ManagedArray;

namespace NumSharp
{
	public static partial class np
	{
        public static NDArray array(Array array, Type dtype = null, int ndim = 1)
        {
            dtype = (dtype == null) ? array.GetType().GetElementType() : dtype;
            
			var nd = new NDArray(dtype);

            if ((array.Rank == 1) && ( !array.GetType().GetElementType().IsArray ))
			{
                nd.Storage = new ManagedArrayEngine(dtype);
                nd.Storage.Allocate(dtype, new Shape(new int[] { array.Length }));

                nd.Storage.SetData(array); 
            }
            else 
            {
                throw new Exception("Method is not implemeneted for multidimensional arrays or jagged arrays.");
            }
            
            return nd;
        }

        /*public static NDArray array(System.Drawing.Bitmap image)
        {
            var imageArray = new NDArray(typeof(Byte));

            var bmpd = image.LockBits(new System.Drawing.Rectangle(0, 0, image.Width, image.Height), System.Drawing.Imaging.ImageLockMode.ReadOnly, image.PixelFormat);
            var dataSize = bmpd.Stride * bmpd.Height;

            var bytes = new byte[dataSize];
            System.Runtime.InteropServices.Marshal.Copy(bmpd.Scan0, bytes, 0, dataSize);
            image.UnlockBits(bmpd);

            imageArray.Storage.Allocate(typeof(byte),new Shape(bmpd.Height, bmpd.Width, System.Drawing.Image.GetPixelFormatSize(image.PixelFormat) / 8),1);
            imageArray.Storage.SetData(bytes);
            
            return imageArray;
        }*/

        public static NDArray array<T>(T[][] data)
        {
            var nd = new NDArray(typeof(T), new Shape(data.Length, data[0].Length));
            
            for (int row = 0; row < data.Length; row++)
            {
                for (int col = 0; col < data[row].Length; col++)
                {
                    nd[row,col] = data[row][col];
                }
            }

            return nd;
        }

        public static NDArray array<T>(T[,] data)
        {
            var nd = new NDArray(typeof(T), new Shape(data.GetLength(0), data.GetLength(1)));

            for (int dim0 = 0; dim0 < data.GetLength(0); dim0++)
            {
                for (int dim1 = 0; dim1 < data.GetLength(1); dim1++)
                {
                    nd[dim0, dim1] = data[dim0, dim1];
                }
            }

            return nd;
        }

        public static NDArray array<T>(T[,,] data)
        {
            var nd = new NDArray(typeof(T), new Shape(data.GetLength(0), data.GetLength(1), data.GetLength(2)));

            for (int dim0 = 0; dim0 < data.GetLength(0); dim0++)
            {
                for (int dim1 = 0; dim1 < data.GetLength(1); dim1++)
                {
                    for (int dim2 = 0; dim2 < data.GetLength(2); dim2++)
                    {
                        nd[dim0,dim1,dim2] = data[dim0, dim1, dim2];
                    }
                }
            }

            return nd;
        }

        public static NDArray array<T>(params T[] data)
        {
            var nd = new NDArray(typeof(T), data.Length);

            nd.Storage.SetData<T>(data);

            return nd;
        }
    }
}
