using System;
using System.Diagnostics.CodeAnalysis;
using System.Drawing;
using System.Drawing.Imaging;
using NumSharp.Backends;
using NumSharp.Backends.Unmanaged;

// ReSharper disable once CheckNamespace
namespace NumSharp
{
    [SuppressMessage("ReSharper", "SuggestVarOrType_SimpleTypes")]
    public static class np_
    {
        /// <summary>
        ///     Creates <see cref="NDArray"/> from given <see cref="Bitmap"/>.
        /// </summary>
        /// <param name="image">The image to load data from.</param>
        /// <param name="flat">
        ///     If true, returns NDArray be 1-d of pixels: `R1G1B1R2G2B2 ... RnGnBn` where n is the amount of pixels in the image.<br></br>
        ///     If false, returns a 4-d NDArray shaped: (1, bmpData.Height, bmpData.Width, 3)
        /// </param>
        /// <param name="copy">
        ///     If true, performs <see cref="Bitmap.LockBits(System.Drawing.Rectangle,System.Drawing.Imaging.ImageLockMode,System.Drawing.Imaging.PixelFormat)"/>
        ///     and then copies the data to a new <see cref="NDArray"/> then finally releasing the locked bits.<br></br>
        ///     If false, It'll call <see cref="Bitmap.LockBits(System.Drawing.Rectangle,System.Drawing.Imaging.ImageLockMode,System.Drawing.Imaging.PixelFormat)"/>
        ///     , wraps the <see cref="BitmapData.Scan0"/> with an NDArray and call <see cref="Bitmap.UnlockBits"/> only when the NDArray will be collected by the <see cref="GC"/>.
        /// </param>
        /// <returns>An NDArray that holds the pixel data of the given bitmap</returns>
        public static unsafe NDArray ToNDArray(this System.Drawing.Bitmap image, bool flat = false, bool copy = true)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));

            BitmapData bmpData = image.LockBits(new Rectangle(0, 0, image.Width, image.Height), ImageLockMode.ReadOnly, image.PixelFormat);
            if (copy)
                try
                {
                    unsafe
                    {
                        //Create a 1d vector without filling it's values to zero (similar to np.empty)
                        var nd = new NDArray(NPTypeCode.Byte, Shape.Vector(bmpData.Stride * image.Height), fillZeros: false);

                        // Get the respective addresses
                        byte* src = (byte*)bmpData.Scan0;
                        byte* dst = (byte*)nd.Unsafe.Address; //we can use unsafe because we just allocated that array and we know for sure it is contagious.

                        // Copy the RGB values into the array.
                        Buffer.MemoryCopy(src, dst, nd.size, nd.size); //faster than Marshal.Copy
                        return nd.reshape(1, image.Height, image.Width, 3);
                    }
                }
                finally
                {
                    image.UnlockBits(bmpData);
                }
            else
            {
                var nd = new NDArray(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)bmpData.Scan0, bmpData.Stride * bmpData.Height, () => image.UnlockBits(bmpData))));
                return flat ? nd : nd.reshape(1, bmpData.Height, bmpData.Width, 3);
            }
        }

        /// <summary>
        ///     Creates <see cref="NDArray"/> from given <see cref="Image"/>.
        /// </summary>
        /// <param name="image">The image to load data from.</param>
        /// <param name="flat">
        ///     If true, returns NDArray be 1-d of pixels: `R1G1B1R2G2B2 ... RnGnBn` where n is the amount of pixels in the image.<br></br>
        ///     If false, returns a 4-d NDArray shaped: (1, bmpData.Height, bmpData.Width, 3)
        /// </param>
        /// <param name="copy">
        ///     If true, performs <see cref="Bitmap.LockBits(System.Drawing.Rectangle,System.Drawing.Imaging.ImageLockMode,System.Drawing.Imaging.PixelFormat)"/>
        ///     and then copies the data to a new <see cref="NDArray"/> then finally releasing the locked bits.<br></br>
        ///     If false, It'll call <see cref="Bitmap.LockBits(System.Drawing.Rectangle,System.Drawing.Imaging.ImageLockMode,System.Drawing.Imaging.PixelFormat)"/>
        ///     , wraps the <see cref="BitmapData.Scan0"/> with an NDArray and call <see cref="Bitmap.UnlockBits"/> only when the NDArray will be collected by the <see cref="GC"/>.
        /// </param>
        /// <returns>An NDArray that holds the pixel data of the given bitmap</returns>
        public static NDArray ToNDArray(this Image image, bool flat = false, bool copy = true) 
            => ToNDArray(new System.Drawing.Bitmap(image), flat, copy);

        /// <summary>
        ///     Wraps given <see cref="BitmapData"/> as n <see cref="NDArray"/> without performing copy.
        /// </summary>
        /// <param name="bmpData">Targetted bitmap data with reading capabilities.</param>
        /// <param name="flat">
        ///     If true, returns NDArray be 1-d of pixels: R1G1B1R2G2B2 ... RnGnBn where n is the amount of pixels in the image.<br></br>
        ///     If false, returns a 4-d NDArray shaped: (1, bmpData.Height, bmpData.Width, 3)
        /// </param>
        /// <returns>An NDArray that wraps the given bitmap, doesn't copy</returns>
        /// <remarks>If the BitmapData is unlocked via <see cref="Bitmap.UnlockBits"/> - the NDArray will point to an invalid address which will cause heap corruption. Use with caution!</remarks>
        public static unsafe NDArray AsNDArray(this BitmapData bmpData, bool flat = false)
        {
            if (bmpData == null)
                throw new ArgumentNullException(nameof(bmpData));

            var nd = new NDArray(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)bmpData.Scan0, bmpData.Stride * bmpData.Height)));
            return flat ? nd : nd.reshape(1, bmpData.Height, bmpData.Width, 3);
        }
    }
}
