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
    [SuppressMessage("ReSharper", "UnusedMember.Global")]
    public static class np_
    {
        /// <summary>
        ///     Creates <see cref="NDArray"/> from given <see cref="Bitmap"/>.
        /// </summary>
        /// <param name="image">The image to load data from.</param>
        /// <param name="flat">
        ///     If true, returns NDArray be 1-d of pixels: `R1G1B1R2G2B2 ... RnGnBn` where n is the amount of pixels in the image.<br></br>
        ///     If false, returns a 4-d NDArray shaped: (1, bmpData.Height, bmpData.Width, bbp)<br></br>
        ///         or (1, bmpData.Height, bmpData.Width, 4) if alpha is present and <paramref name="discardAlpha"/> is false.
        /// </param>
        /// <param name="copy">
        ///     If true, performs <see cref="Bitmap.LockBits(System.Drawing.Rectangle,System.Drawing.Imaging.ImageLockMode,System.Drawing.Imaging.PixelFormat)"/>
        ///     and then copies the data to a new <see cref="NDArray"/> then finally releasing the locked bits.<br></br>
        ///     If false, It'll call <see cref="Bitmap.LockBits(System.Drawing.Rectangle,System.Drawing.Imaging.ImageLockMode,System.Drawing.Imaging.PixelFormat)"/>
        ///     , wraps the <see cref="BitmapData.Scan0"/> with an NDArray and call <see cref="Bitmap.UnlockBits"/> only when the NDArray will be collected by the <see cref="GC"/>.
        /// </param>
        /// <param name="discardAlpha">If the given <see cref="Bitmap"/> has an alpha pixel (transparency pixel), discard that data or return a slice without the alpha (depends on <paramref name="copy"/>).</param>
        /// <returns>An NDArray that holds the pixel data of the given bitmap</returns>
        [SuppressMessage("ReSharper", "SuggestVarOrType_Elsewhere")]
        public static unsafe NDArray ToNDArray(this System.Drawing.Bitmap image, bool flat = false, bool copy = true, bool discardAlpha = false)
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
                        Buffer.MemoryCopy(src, dst, bmpData.Stride * image.Height, nd.size); //faster than Marshal.Copy
                        var ret = nd.reshape(1, image.Height, image.Width, bmpData.Stride / bmpData.Width);
                        if (discardAlpha && ret.shape[3] == 4)
                            ret = ret[Slice.All, Slice.All, Slice.All, new Slice(stop: 3)];

                        return flat && ret.ndim != 1 ? ret.flat : ret;
                    }
                }
                finally
                {
                    try
                    {
                        image.UnlockBits(bmpData);
                    }
                    catch (ArgumentException)
                    {
                        //swallow
                    }
                }
            else
            {
                var ret = new NDArray(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)bmpData.Scan0, bmpData.Stride * bmpData.Height, () =>
                {
                    try
                    {
                        image.UnlockBits(bmpData);
                    }
                    catch (ArgumentException)
                    {
                        //swallow
                    }
                })));

                if (flat)
                {
                    if (discardAlpha)
                    {
                        if (bmpData.Stride / bmpData.Width == 4) //1byte-per-color
                        {
                            return ReshapeFlatData(ret, bmpData) // reshape
                                       [Slice.All, Slice.All, Slice.All, new Slice(stop: 3)] //slice
                                      .flat; //flatten
                        }

                        if (bmpData.Stride / bmpData.Width == 8) //2bytes-per-color
                        {
                            return ReshapeFlatData(ret, bmpData) // reshape
                                       [Slice.All, Slice.All, Slice.All, new Slice(stop: 6)] //slice
                                      .flat; //flatten
                        }

                        throw new NotSupportedException($"Given bbp ({bmpData.Stride / bmpData.Width}) is not supported.");
                    }

                    return ret.flat;
                }
                else
                {
                    ret = ReshapeFlatData(ret, bmpData); //reshape
                    if (discardAlpha)
                    {
                        if (ret.shape[3] == 4) //1byte-per-color
                            ret = ret[Slice.All, Slice.All, Slice.All, new Slice(stop: 3)]; //slice
                        else if (ret.shape[3] == 8) //2bytes-per-color
                            ret = ret[Slice.All, Slice.All, Slice.All, new Slice(stop: 6)]; //slice
                        else
                            throw new NotSupportedException($"Given bbp ({bmpData.Stride / bmpData.Width}) is not supported.");
                    }

                    return ret;
                }
            }
        }

        /// <summary>
        /// Reshapes the flat data to match the size of the bitmap.
        /// </summary>
        /// <param name="ret">flat 1-dimensional array containing the bitmap data</param>
        /// <param name="bmpData">Source bitmap</param>
        /// <returns>An 4-dimensional NDArray that holds the bitmap data contained in `ret`</returns>
        private static NDArray ReshapeFlatData(NDArray ret, BitmapData bmpData)
        {
            var colorBytes = bmpData.Stride / bmpData.Width;
            var strideWidth = bmpData.Stride / colorBytes; // For odd widths, this width is not equal to bmpData.Width
            ret = ret.reshape(1, bmpData.Height, strideWidth, bmpData.Stride / bmpData.Width);
            if (strideWidth != bmpData.Width)
            {
                ret = ret[Slice.All, Slice.All, new Slice(stop: bmpData.Width), new Slice(stop: colorBytes)];
            }

            return ret;
        }

        /// <summary>
        ///     Creates <see cref="NDArray"/> from given <see cref="Image"/>.
        /// </summary>
        /// <param name="image">The image to load data from.</param>
        /// <param name="flat">
        ///     If true, returns NDArray be 1-d of pixels: `R1G1B1R2G2B2 ... RnGnBn` where n is the amount of pixels in the image.<br></br>
        ///     If false, returns a 4-d NDArray shaped: (1, bmpData.Height, bmpData.Width, bbp)
        /// </param>
        /// <param name="copy">
        ///     If true, performs <see cref="Bitmap.LockBits(System.Drawing.Rectangle,System.Drawing.Imaging.ImageLockMode,System.Drawing.Imaging.PixelFormat)"/>
        ///     and then copies the data to a new <see cref="NDArray"/> then finally releasing the locked bits.<br></br>
        ///     If false, It'll call <see cref="Bitmap.LockBits(System.Drawing.Rectangle,System.Drawing.Imaging.ImageLockMode,System.Drawing.Imaging.PixelFormat)"/>
        ///     , wraps the <see cref="BitmapData.Scan0"/> with an NDArray and call <see cref="Bitmap.UnlockBits"/> only when the NDArray will be collected by the <see cref="GC"/>.
        /// </param>
        /// <param name="discardAlpha">If the given <see cref="Bitmap"/> has an alpha pixel (transparency pixel), discard that data or return a slice without the alpha (depends on <paramref name="copy"/>).</param>
        /// <returns>An NDArray that holds the pixel data of the given bitmap</returns>
        public static NDArray ToNDArray(this Image image, bool flat = false, bool copy = true, bool discardAlpha = false)
        {
            if (image == null)
                throw new ArgumentNullException(nameof(image));

            return ToNDArray(new System.Drawing.Bitmap(image), flat, copy, discardAlpha);
        }

        /// <summary>
        ///     Wraps given <see cref="BitmapData"/> as n <see cref="NDArray"/> without performing copy.
        /// </summary>
        /// <param name="bmpData">Targetted bitmap data with reading capabilities.</param>
        /// <param name="flat">
        ///     If true, returns NDArray be 1-d of pixels: R1G1B1R2G2B2 ... RnGnBn where n is the amount of pixels in the image.<br></br>
        ///     If false, returns a 4-d NDArray shaped: (1, bmpData.Height, bmpData.Width, bbp)
        /// </param>
        /// <param name="discardAlpha">If the given <see cref="Bitmap"/> has an alpha pixel (transparency pixel), return a slice without the alpha.</param>
        /// <returns>An NDArray that wraps the given bitmap, doesn't copy</returns>
        /// <remarks>If the BitmapData is unlocked via <see cref="Bitmap.UnlockBits"/> - the NDArray will point to an invalid address which will cause heap corruption. Use with caution!</remarks>
        public static unsafe NDArray AsNDArray(this BitmapData bmpData, bool flat = false, bool discardAlpha = false)
        {
            if (bmpData == null)
                throw new ArgumentNullException(nameof(bmpData));

            var ret = new NDArray(new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)bmpData.Scan0, bmpData.Stride * bmpData.Height)));
            if (flat)
            {
                if (ret.shape[3] == 4 && discardAlpha)
                {
                    return ret.reshape(1, bmpData.Height, bmpData.Width, bmpData.Stride / bmpData.Width) //reshape
                               [Slice.All, Slice.All, Slice.All, new Slice(stop: 3)] //slice
                              .flat; //flatten
                }

                return ret;
            }
            else
            {
                ret = ret.reshape(1, bmpData.Height, bmpData.Width, bmpData.Stride / bmpData.Width); //reshape

                if (ret.shape[3] == 4 && discardAlpha)
                    ret = ret[Slice.All, Slice.All, Slice.All, new Slice(stop: 3)]; //slice

                return ret;
            }
        }

        /// <summary>
        ///     Converts <see cref="NDArray"/> to a <see cref="Bitmap"/>. 
        /// </summary>
        /// <param name="nd">The <see cref="NDArray"/> to copy pixels from, <see cref="Shape"/> is ignored completely. If nd.Unsafe.Shape.IsContiguous == false then a copy is made.</param>
        /// <param name="width">The height of the <see cref="Bitmap"/></param>
        /// <param name="height">The width of the <see cref="Bitmap"/></param>
        /// <param name="format">The format of the expected bitmap, Must be matching to given NDArray otherwise unexpected results might occur.</param>
        /// <returns>A <see cref="Bitmap"/></returns>
        /// <exception cref="ArgumentException">When nd.size != width*height, which means the ndarray be turned into the given bitmap size.</exception>
        public static unsafe Bitmap ToBitmap(this NDArray nd, int width, int height, PixelFormat format = PixelFormat.DontCare)
        {
            if (nd == null)
                throw new ArgumentNullException(nameof(nd));

            //if flat then initialize based on given format
            if (nd.ndim == 1 && format != PixelFormat.DontCare)
                nd = nd.reshape(1, height, width, format.ToBytesPerPixel()); //theres a check internally for size mismatch.

            if (nd.ndim != 4)
                throw new ArgumentException("ndarray was expected to be of 4-dimensions, (1, bmpData.Height, bmpData.Width, bytesPerPixel)");

            if (nd.shape[0] != 1)
                throw new ArgumentException($"ndarray has more than one picture in it ({nd.shape[0]}) based on the first dimension.");

            var bbp = nd.shape[3]; //bytes per pixel.
            if (bbp != extractFormatNumber())
                throw new ArgumentException($"Given PixelFormat: {format} does not match the number of bytes per pixel in the 4th dimension of given ndarray.");

            if (bbp * width * height != nd.size)
                throw new ArgumentException($"The expected size does not match the size of given ndarray. (expected: {bbp * width * height}, actual: {nd.size})");

            var ret = new Bitmap(width, height, format);
            var bitdata = ret.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, format);
            try
            {
                var dst = new ArraySlice<byte>(new UnmanagedMemoryBlock<byte>((byte*)bitdata.Scan0, bitdata.Stride * bitdata.Height));
                if (nd.Shape.IsContiguous)
                    nd.CopyTo(dst);
                else
                    MultiIterator.Assign(new UnmanagedStorage(dst, Shape.Vector(bitdata.Stride * bitdata.Height)), nd.Unsafe.Storage);
            }
            finally
            {
                try
                {
                    ret.UnlockBits(bitdata);
                }
                catch (ArgumentException)
                {
                    //swallow
                }
            }

            return ret;

            int extractFormatNumber()
            {
                if (format == PixelFormat.DontCare)
                {
                    switch (bbp)
                    {
                        case 3:
                            format = PixelFormat.Format24bppRgb;
                            break;
                        case 4:
                            format = PixelFormat.Format32bppArgb;
                            break;
                        case 6:
                            format = PixelFormat.Format48bppRgb;
                            break;
                        case 8:
                            format = PixelFormat.Format64bppArgb;
                            break;
                    }

                    return bbp;
                }

                return format.ToBytesPerPixel();
            }
        }

        /// <summary>
        ///     Converts <see cref="NDArray"/> to a <see cref="Bitmap"/>. 
        /// </summary>
        /// <param name="nd">The <see cref="NDArray"/> to copy pixels from, <see cref="Shape"/> is ignored completely. If nd.Unsafe.Shape.IsContiguous == false then a copy is made.</param>
        /// <param name="format">The format of the expected bitmap, Must be matching to given NDArray otherwise unexpected results might occur.</param>
        /// <returns>A <see cref="Bitmap"/></returns>
        /// <exception cref="ArgumentException">When nd.size != width*height, which means the ndarray be turned into the given bitmap size.</exception>
        public static Bitmap ToBitmap(this NDArray nd, PixelFormat format = PixelFormat.DontCare)
        {
            if (nd == null)
                throw new ArgumentNullException(nameof(nd));
            if (nd.ndim != 4)
                throw new ArgumentException("ndarray was expected to be of 4-dimensions, (1, bmpData.Height, bmpData.Width, bytesPerPixel)");
            if (nd.shape[0] != 1)
                throw new ArgumentException($"ndarray has more than one picture in it ({nd.shape[0]}) based on the first dimension.");

            var height = nd.shape[1];
            var width = nd.shape[2];

            return ToBitmap(nd, width, height, format);
        }

        /// <summary>
        ///     Returns how many bytes are per pixel based on given <paramref name="format"/>.
        /// </summary>
        /// <exception cref="ArgumentException">The format is not supported by us.</exception>
        /// <exception cref="ArgumentOutOfRangeException">Invalid PixelFormat enum value.</exception>
        public static int ToBytesPerPixel(this PixelFormat format)
        {
            int ret;
            switch (format)
            {
                case PixelFormat.Format16bppArgb1555:
                    ret = 16;
                    break;
                case PixelFormat.Format16bppGrayScale:
                    ret = 16;
                    break;
                case PixelFormat.Format16bppRgb555:
                    ret = 16;
                    break;
                case PixelFormat.Format16bppRgb565:
                    ret = 16;
                    break;
                case PixelFormat.Format24bppRgb:
                    ret = 24;
                    break;
                case PixelFormat.Format32bppArgb:
                    ret = 32;
                    break;
                case PixelFormat.Format32bppPArgb:
                    ret = 32;
                    break;
                case PixelFormat.Format32bppRgb:
                    ret = 32;
                    break;
                case PixelFormat.Format48bppRgb:
                    ret = 48;
                    break;
                case PixelFormat.Format64bppArgb:
                    ret = 64;
                    break;
                case PixelFormat.Format64bppPArgb:
                    ret = 64;
                    break;
                case PixelFormat.Format1bppIndexed:
                case PixelFormat.Format8bppIndexed:
                case PixelFormat.Format4bppIndexed:
                case PixelFormat.Alpha:
                case PixelFormat.Canonical:
                case PixelFormat.Extended:
                case PixelFormat.Gdi:
                case PixelFormat.Indexed:
                case PixelFormat.Max:
                case PixelFormat.PAlpha:
                    throw new ArgumentException($"Given PixelFormat: {format} is not supported.");
                case PixelFormat.DontCare:
                    throw new ArgumentException("Given PixelFormat can't be DontCare.");
                default:
                    throw new ArgumentOutOfRangeException(nameof(format), format, null);
            }

            return ret / 8;
        }
    }
}
