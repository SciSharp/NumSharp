using System.Drawing.Imaging;
using System.Resources;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest
{
    [TestClass]
    public class BitmapWithAlphaTests : TestClass
    {
        [TestMethod]
        public void ToNDArray_Case1()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(discardAlpha: true, copy: true);
            nd.Should().BeShaped(1, 165, 400, 3);
        }

        [TestMethod]
        public void ToNDArray_Case2()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(discardAlpha: true, copy: false);
            nd.Should().BeShaped(1, 165, 400, 3);
        }

        [TestMethod]
        public void ToNDArray_Case3()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(discardAlpha: false, copy: true);
            nd.Should().BeShaped(1, 165, 400, 4);
        }

        [TestMethod]
        public void ToNDArray_Case4()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(discardAlpha: false, copy: false);
            nd.Should().BeShaped(1, 165, 400, 4);
        }
        
        
        [TestMethod]
        public void ToNDArray_Odd_Width()
        {
            var bitmap = EmbeddedBitmap("odd-width");
            var nd = bitmap.ToNDArray(discardAlpha: false, copy: false);
            nd.Should().BeShaped(1, 554, 475, 3);
        }

        [TestMethod]
        public void ToNDArray_Case5_flat()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(discardAlpha: true, copy: true, flat: true);
            nd.Should().BeShaped(1 * 165 * 400 * 3);
        }

        [TestMethod]
        public void ToNDArray_Case6_flat()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(discardAlpha: true, copy: false, flat: true);
            nd.Should().BeShaped(1 * 165 * 400 * 3);
        }

        [TestMethod]
        public void ToNDArray_Case7_flat()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(discardAlpha: false, copy: true, flat: true);
            nd.Should().BeShaped(1 * 165 * 400 * 4);
        }

        [TestMethod]
        public void ToNDArray_Case8_flat()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(discardAlpha: false, copy: false, flat: true);
            nd.Should().BeShaped(1 * 165 * 400 * 4);
        }

        [TestMethod]
        public void ToBitmap_PixelFormat_DontCare_Case1()
        {
            var bm = np.arange(0, 3 * 3 * 3).reshape(1, 3, 3, 3).astype(NPTypeCode.Byte).ToBitmap();
            bm.PixelFormat.Should().Be(PixelFormat.Format24bppRgb);
        }

        [TestMethod]
        public void ToBitmap_PixelFormat_DontCare_Case2()
        {
            var bm = np.arange(0, 3 * 3 * 4).reshape(1, 3, 3, 4).astype(NPTypeCode.Byte).ToBitmap();
            bm.PixelFormat.Should().Be(PixelFormat.Format32bppArgb);
        }
    }
}
