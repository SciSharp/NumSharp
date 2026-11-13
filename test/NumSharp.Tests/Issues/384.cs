using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;

namespace NumSharp.Tests.Issues
{
    /// <summary>
    ///     Backs the answer that closed GitHub #384, "Save NDArray as png image" (https://github.com/SciSharp/NumSharp/issues/384):
    ///     with NumSharp.Bitmap a (1, H, W, 3|4) uint8 array becomes a <c>Bitmap</c>, saves as PNG, and reads back with
    ///     <c>ToNDArray(copy: true)</c>; the 32bpp round trip is exact.
    /// </summary>
    [TestClass]
    [WindowsOnly]
    public class Issue384
    {
        /// <summary>
        ///     The answer's three lines: <c>nd.ToBitmap()</c>, <c>bmp.Save("out.png", ImageFormat.Png)</c>,
        ///     <c>new Bitmap("out.png").ToNDArray(copy: true)</c> give back the same (1, 4, 6, 4) uint8 array.
        /// </summary>
        [TestMethod]
        public void SaveAsPng_AndReadBack_IsExactAt32bpp()
        {
            var nd = np.arange(1 * 4 * 6 * 4).astype(np.uint8).reshape(1, 4, 6, 4);
            var dir = Path.Combine(Path.GetTempPath(), "numsharp-issue384-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(dir);
            try
            {
                var png = Path.Combine(dir, "out.png");
                using (var bmp = nd.ToBitmap())
                {
                    bmp.PixelFormat.Should().Be(PixelFormat.Format32bppArgb);
                    bmp.Save(png, ImageFormat.Png);
                }

                using var loaded = new Bitmap(png);
                var back = loaded.ToNDArray(copy: true);

                back.shape.Should().Equal(1L, 4L, 6L, 4L);
                back.typecode.Should().Be(NPTypeCode.Byte);
                np.array_equal(back, nd).Should().BeTrue("the 32bpp PNG round trip is exact");
            }
            finally
            {
                Directory.Delete(dir, recursive: true);
            }
        }
    }
}
