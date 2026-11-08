using System.Drawing;
using System.Drawing.Imaging;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #406, "C# --> convert image to NDarray"
    ///     (https://github.com/SciSharp/NumSharp/issues/406): NumSharp.Bitmap's <c>image.ToNDArray(copy: true)</c> turns a
    ///     <c>System.Drawing.Image</c> into a (1, H, W, channels) uint8 array.
    /// </summary>
    [TestClass]
    [WindowsOnly]
    public class Issue406
    {
        /// <summary>
        ///     A 32bpp image passed as <c>System.Drawing.Image</c> gives a (1, 2, 3, 4) uint8 array whose pixels read back in
        ///     GDI+'s BGRA order.
        /// </summary>
        [TestMethod]
        public void ImageToNDArray_GivesBatchHeightWidthChannels()
        {
            using var bmp = new Bitmap(3, 2, PixelFormat.Format32bppArgb);
            bmp.SetPixel(2, 1, Color.FromArgb(255, 10, 20, 30));
            Image image = bmp;

            NDArray nd = image.ToNDArray(copy: true);

            nd.shape.Should().Equal(1L, 2L, 3L, 4L);
            nd.typecode.Should().Be(NPTypeCode.Byte);
            nd.GetByte(0, 1, 2, 0).Should().Be(30, "blue comes first in GDI+'s BGRA layout");
            nd.GetByte(0, 1, 2, 1).Should().Be(20);
            nd.GetByte(0, 1, 2, 2).Should().Be(10);
            nd.GetByte(0, 1, 2, 3).Should().Be(255);
        }
    }
}
