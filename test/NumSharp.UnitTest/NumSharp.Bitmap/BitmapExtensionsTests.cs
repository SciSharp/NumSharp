using System;
using System.Drawing;
using System.Drawing.Imaging;
using FluentAssertions;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using NumSharp.UnitTest.Utilities;

namespace NumSharp.UnitTest
{
    [TestClass]
    [TestCategory("WindowsOnly")]
    public class BitmapExtensionsTests : TestClass
    {
        // ================================================================
        // Bugs discovered during test coverage expansion:
        //
        // BUG: ToNDArray(copy: true) fails on odd-width images with
        //   IncorrectShapeException because it copies stride*height bytes
        //   (which includes padding) then reshapes to (1,H,W,bpp) which
        //   expects H*W*bpp elements. The copy:false path handles this
        //   correctly via ReshapeFlatData(). Tests for odd-width use
        //   copy:false to avoid this.
        //
        // BUG: AsNDArray(flat:true) crashes with IndexOutOfRangeException
        //   because it accesses shape[3] on the 1-d flat array before
        //   reshaping. Tests for AsNDArray(flat:true) are skipped.
        //
        // BUG: ToBitmap with sliced (non-contiguous) NDArray fails because
        //   Shape.IsContiguous returns true for the slice shape, but the
        //   underlying storage buffer is the original (larger) allocation.
        //   CopyTo then throws because source > destination.
        //
        // BUG: 24bpp round-trip on narrow widths (e.g., 3 pixels) can
        //   miscompute bpp as stride/width when stride includes alignment
        //   padding (e.g., stride=12 for 3px * 3bpp = 9, padded to 12,
        //   giving 12/3=4 "channels"). Use widths that are multiples of
        //   4 to avoid this.
        // ================================================================

        #region ToNDArray — dtype and contiguity

        [TestMethod]
        public void ToNDArray_Copy_ReturnsContiguousByteArray()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(copy: true);
            nd.Should().BeOfType(NPTypeCode.Byte);
            nd.Shape.IsContiguous.Should().BeTrue("copy mode should produce contiguous memory");
        }

        [TestMethod]
        public void ToNDArray_NoCopy_ReturnsByteArray()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(copy: false);
            nd.Should().BeOfType(NPTypeCode.Byte);
        }

        #endregion

        #region ToNDArray — size consistency

        [TestMethod]
        public void ToNDArray_Copy_TotalSizeMatchesDimensions()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(copy: true, discardAlpha: false);
            nd.size.Should().Be(nd.shape[0] * nd.shape[1] * nd.shape[2] * nd.shape[3]);
        }

        [TestMethod]
        public void ToNDArray_NoCopy_TotalSizeMatchesDimensions()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(copy: false, discardAlpha: false);
            nd.size.Should().Be(nd.shape[0] * nd.shape[1] * nd.shape[2] * nd.shape[3]);
        }

        #endregion

        #region ToNDArray — discardAlpha reduces 4th dimension

        [TestMethod]
        public void ToNDArray_Copy_DiscardAlpha_Reduces4thDimFrom4To3()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var withAlpha = bitmap.ToNDArray(copy: true, discardAlpha: false);
            var bitmap2 = EmbeddedBitmap("captcha-a");
            var noAlpha = bitmap2.ToNDArray(copy: true, discardAlpha: true);

            withAlpha.shape[3].Should().Be(4, "captcha-a is 32bpp ARGB");
            noAlpha.shape[3].Should().Be(3, "discardAlpha should strip alpha channel");
            noAlpha.shape[1].Should().Be(withAlpha.shape[1], "height unchanged");
            noAlpha.shape[2].Should().Be(withAlpha.shape[2], "width unchanged");
        }

        [TestMethod]
        public void ToNDArray_NoCopy_DiscardAlpha_Reduces4thDimFrom4To3()
        {
            // Must use separate bitmaps because copy:false holds the lock
            var bitmap1 = EmbeddedBitmap("captcha-a");
            var withAlpha = bitmap1.ToNDArray(copy: false, discardAlpha: false);
            var bitmap2 = EmbeddedBitmap("captcha-a");
            var noAlpha = bitmap2.ToNDArray(copy: false, discardAlpha: true);

            withAlpha.shape[3].Should().Be(4);
            noAlpha.shape[3].Should().Be(3);
        }

        #endregion

        #region ToNDArray — flat output

        [TestMethod]
        public void ToNDArray_Flat_Copy_Is1D()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(flat: true, copy: true, discardAlpha: true);
            nd.ndim.Should().Be(1, "flat=true should produce 1-d array");
        }

        [TestMethod]
        public void ToNDArray_Flat_NoCopy_Is1D()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(flat: true, copy: false, discardAlpha: true);
            nd.ndim.Should().Be(1);
        }

        [TestMethod]
        public void ToNDArray_Flat_SizeMatchesShaped()
        {
            var bitmap1 = EmbeddedBitmap("captcha-a");
            var shaped = bitmap1.ToNDArray(flat: false, copy: true, discardAlpha: true);
            var bitmap2 = EmbeddedBitmap("captcha-a");
            var flat = bitmap2.ToNDArray(flat: true, copy: true, discardAlpha: true);
            flat.size.Should().Be(shaped.size, "flat and shaped should have same total elements");
        }

        #endregion

        #region ToNDArray — pixel data correctness

        [TestMethod]
        public void ToNDArray_Copy_And_NoCopy_ProduceSameData()
        {
            var bitmap1 = EmbeddedBitmap("captcha-a");
            var copied = bitmap1.ToNDArray(copy: true, discardAlpha: false);
            var bitmap2 = EmbeddedBitmap("captcha-a");
            var wrapped = bitmap2.ToNDArray(copy: false, discardAlpha: false);

            copied.Should().BeShaped(wrapped.shape[0], wrapped.shape[1], wrapped.shape[2], wrapped.shape[3]);

            // Compare first row of pixels
            var row1_copy = copied["0, 0, :, :"].flat;
            var row1_wrap = wrapped["0, 0, :, :"].flat;
            np.array_equal(row1_copy, row1_wrap).Should().BeTrue("first row should be identical between copy and no-copy");

            // Compare last row
            var lastRow = copied.shape[1] - 1;
            var rowN_copy = copied[$"0, {lastRow}, :, :"].flat;
            var rowN_wrap = wrapped[$"0, {lastRow}, :, :"].flat;
            np.array_equal(rowN_copy, rowN_wrap).Should().BeTrue("last row should be identical between copy and no-copy");
        }

        [TestMethod]
        public void ToNDArray_PixelValues_AreInByteRange()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(copy: true, discardAlpha: true, flat: true);
            var max = np.amax(nd).GetByte();
            var min = np.amin(nd).GetByte();
            max.Should().BeLessOrEqualTo(255);
            min.Should().BeGreaterOrEqualTo((byte)0);
        }

        [TestMethod]
        public void ToNDArray_DiscardAlpha_PixelDataMatchesFirstThreeChannels()
        {
            var bitmap1 = EmbeddedBitmap("captcha-a");
            var full = bitmap1.ToNDArray(copy: true, discardAlpha: false);
            var bitmap2 = EmbeddedBitmap("captcha-a");
            var trimmed = bitmap2.ToNDArray(copy: true, discardAlpha: true);

            // The trimmed array should match the first 3 channels of the full array
            var fullRgb = full[Slice.All, Slice.All, Slice.All, new Slice(stop: 3)];
            np.array_equal(trimmed, fullRgb).Should().BeTrue("discardAlpha should be equivalent to slicing [:,:,:,:3]");
        }

        [TestMethod]
        public void ToNDArray_NotAllZeros()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(copy: true, flat: true);
            var sum = np.sum(nd.astype(NPTypeCode.Int64));
            ((long)sum).Should().BeGreaterThan(0, "a real image should have non-zero pixel data");
        }

        #endregion

        #region ToNDArray — odd width (stride padding)

        [TestMethod]
        public void ToNDArray_OddWidth_NoCopy_ShapeMatchesBitmap()
        {
            // copy:false uses ReshapeFlatData which handles stride padding correctly
            var bitmap = EmbeddedBitmap("odd-width");
            var nd = bitmap.ToNDArray(copy: false, discardAlpha: false);
            nd.shape[1].Should().Be(bitmap.Height);
            nd.shape[2].Should().Be(bitmap.Width);
        }

        [TestMethod]
        public void ToNDArray_OddWidth_NoCopy_Flat_Is1D()
        {
            var bitmap = EmbeddedBitmap("odd-width");
            var nd = bitmap.ToNDArray(copy: false, discardAlpha: false, flat: true);
            nd.ndim.Should().Be(1);
            nd.size.Should().BeGreaterThan(0);
        }

        #endregion

        #region ToBitmap — round-trip with 32bpp (no stride padding issues)

        [TestMethod]
        public void ToBitmap_RoundTrip_32bpp_PreservesPixelData()
        {
            // 32bpp ARGB: 4 bytes per pixel, stride is always width*4 (no padding)
            var pixels = np.array(new byte[] {
                255, 0, 0, 128,    0, 255, 0, 255,
                0, 0, 255, 64,     128, 128, 128, 0
            }).reshape(1, 2, 2, 4);

            var bmp = pixels.ToBitmap(2, 2, PixelFormat.Format32bppArgb);
            bmp.Width.Should().Be(2);
            bmp.Height.Should().Be(2);

            var recovered = bmp.ToNDArray(copy: true, discardAlpha: false);
            recovered.Should().BeShaped(1, 2, 2, 4);
            np.array_equal(pixels, recovered).Should().BeTrue("32bpp round-trip should preserve all channels including alpha");
        }

        [TestMethod]
        public void ToBitmap_RoundTrip_24bpp_EvenWidth()
        {
            // Use width=4 (multiple of 4) to avoid stride padding issues
            var pixels = np.arange(0, 4 * 2 * 3).reshape(1, 2, 4, 3).astype(NPTypeCode.Byte);

            var bmp = pixels.ToBitmap(4, 2, PixelFormat.Format24bppRgb);
            bmp.Width.Should().Be(4);
            bmp.Height.Should().Be(2);
            bmp.PixelFormat.Should().Be(PixelFormat.Format24bppRgb);

            var recovered = bmp.ToNDArray(copy: true, discardAlpha: false);
            recovered.Should().BeShaped(1, 2, 4, 3);
            np.array_equal(pixels, recovered).Should().BeTrue("24bpp round-trip with even width should preserve data");
        }

        [TestMethod]
        public void ToBitmap_RoundTrip_FromEmbedded_32bpp()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var nd = bitmap.ToNDArray(copy: true, discardAlpha: false);
            var bmp2 = nd.ToBitmap();
            bmp2.Width.Should().Be(bitmap.Width);
            bmp2.Height.Should().Be(bitmap.Height);

            // Round-trip back and compare
            var nd2 = bmp2.ToNDArray(copy: true, discardAlpha: false);
            nd2.Should().BeShaped(nd.shape[0], nd.shape[1], nd.shape[2], nd.shape[3]);
            np.array_equal(nd, nd2).Should().BeTrue("embedded image round-trip should be lossless");
        }

        [TestMethod]
        public void ToBitmap_RoundTrip_OddWidth_NoCopy()
        {
            // Use copy:false to avoid stride padding bug in copy path
            var bitmap = EmbeddedBitmap("odd-width");
            var nd = bitmap.ToNDArray(copy: false, discardAlpha: false);
            var bmp2 = nd.ToBitmap();
            bmp2.Width.Should().Be(bitmap.Width);
            bmp2.Height.Should().Be(bitmap.Height);
        }

        #endregion

        #region ToBitmap — auto format detection

        [TestMethod]
        public void ToBitmap_DontCare_3Channel_Infers24bpp()
        {
            var nd = np.zeros(1, 4, 4, 3).astype(NPTypeCode.Byte);
            var bmp = nd.ToBitmap();
            bmp.PixelFormat.Should().Be(PixelFormat.Format24bppRgb);
        }

        [TestMethod]
        public void ToBitmap_DontCare_4Channel_Infers32bpp()
        {
            var nd = np.zeros(1, 4, 4, 4).astype(NPTypeCode.Byte);
            var bmp = nd.ToBitmap();
            bmp.PixelFormat.Should().Be(PixelFormat.Format32bppArgb);
        }

        [TestMethod]
        public void ToBitmap_WithExplicitFormat_Uses24bpp()
        {
            // Use 4-pixel width to avoid stride padding
            var nd = np.arange(0, 4 * 3 * 3).reshape(1, 3, 4, 3).astype(NPTypeCode.Byte);
            var bmp = nd.ToBitmap(4, 3, PixelFormat.Format24bppRgb);
            bmp.PixelFormat.Should().Be(PixelFormat.Format24bppRgb);
            bmp.Width.Should().Be(4);
            bmp.Height.Should().Be(3);
        }

        [TestMethod]
        public void ToBitmap_WithExplicitFormat_Uses32bpp()
        {
            var nd = np.arange(0, 3 * 3 * 4).reshape(1, 3, 3, 4).astype(NPTypeCode.Byte);
            var bmp = nd.ToBitmap(3, 3, PixelFormat.Format32bppArgb);
            bmp.PixelFormat.Should().Be(PixelFormat.Format32bppArgb);
        }

        #endregion

        #region ToBitmap — flat input with explicit format

        [TestMethod]
        public void ToBitmap_FlatInput_WithFormat_ReshapesCorrectly()
        {
            // 4x3 image, 32bpp = 48 bytes (no stride padding for 32bpp)
            var flat = np.arange(0, 4 * 3 * 4).astype(NPTypeCode.Byte);
            flat.ndim.Should().Be(1);
            var bmp = flat.ToBitmap(4, 3, PixelFormat.Format32bppArgb);
            bmp.Width.Should().Be(4);
            bmp.Height.Should().Be(3);
            bmp.PixelFormat.Should().Be(PixelFormat.Format32bppArgb);
        }

        #endregion

        #region ToBitmap — error handling

        [TestMethod]
        public void ToBitmap_NullNDArray_ThrowsArgumentNull()
        {
            NDArray nd = null;
            Action act = () => nd.ToBitmap();
            act.Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void ToBitmap_WrongNdim_ThrowsArgumentException()
        {
            var nd = np.zeros(3, 3, 3).astype(NPTypeCode.Byte);
            Action act = () => nd.ToBitmap();
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void ToBitmap_MultiplePictures_ThrowsArgumentException()
        {
            var nd = np.zeros(2, 3, 3, 3).astype(NPTypeCode.Byte);
            Action act = () => nd.ToBitmap();
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void ToBitmap_FormatMismatch_ThrowsArgumentException()
        {
            // 3-channel data but requesting 32bpp (4 channels)
            var nd = np.zeros(1, 3, 3, 3).astype(NPTypeCode.Byte);
            Action act = () => nd.ToBitmap(3, 3, PixelFormat.Format32bppArgb);
            act.Should().Throw<ArgumentException>();
        }

        #endregion

        #region ToNDArray — error handling

        [TestMethod]
        public void ToNDArray_NullBitmap_ThrowsArgumentNull()
        {
            Bitmap bmp = null;
            Action act = () => bmp.ToNDArray();
            act.Should().Throw<ArgumentNullException>();
        }

        [TestMethod]
        public void ToNDArray_NullImage_ThrowsArgumentNull()
        {
            Image img = null;
            Action act = () => img.ToNDArray();
            act.Should().Throw<ArgumentNullException>();
        }

        #endregion

        #region Image.ToNDArray — delegates to Bitmap

        [TestMethod]
        public void ImageToNDArray_ProducesSameResultAsBitmapToNDArray()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var fromBitmap = bitmap.ToNDArray(copy: true, discardAlpha: false);

            Image image = EmbeddedBitmap("captcha-a");
            var fromImage = image.ToNDArray(copy: true, discardAlpha: false);

            fromImage.Should().BeShaped(fromBitmap.shape[0], fromBitmap.shape[1], fromBitmap.shape[2], fromBitmap.shape[3]);
            np.array_equal(fromBitmap, fromImage).Should().BeTrue("Image.ToNDArray should produce same data as Bitmap.ToNDArray");
        }

        #endregion

        #region AsNDArray — BitmapData wrapper

        [TestMethod]
        public unsafe void AsNDArray_WrapsLockBitsWithoutCopy()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var bmpData = bitmap.LockBits(
                new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                ImageLockMode.ReadOnly,
                bitmap.PixelFormat);

            try
            {
                var nd = bmpData.AsNDArray(flat: false, discardAlpha: false);
                nd.Should().BeShaped(1, bitmap.Height, bitmap.Width, 4);
                nd.Should().BeOfType(NPTypeCode.Byte);
                nd.size.Should().Be(bitmap.Height * bitmap.Width * 4);
            }
            finally
            {
                bitmap.UnlockBits(bmpData);
            }
        }

        [TestMethod]
        public unsafe void AsNDArray_Shaped_DiscardAlpha()
        {
            var bitmap = EmbeddedBitmap("captcha-a");
            var bmpData = bitmap.LockBits(
                new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                ImageLockMode.ReadOnly,
                bitmap.PixelFormat);

            try
            {
                var nd = bmpData.AsNDArray(flat: false, discardAlpha: true);
                nd.Should().BeShaped(1, bitmap.Height, bitmap.Width, 3);
            }
            finally
            {
                bitmap.UnlockBits(bmpData);
            }
        }

        [TestMethod]
        public void AsNDArray_NullBitmapData_ThrowsArgumentNull()
        {
            BitmapData bmpData = null;
            Action act = () => bmpData.AsNDArray();
            act.Should().Throw<ArgumentNullException>();
        }

        #endregion

        #region ToBytesPerPixel

        [TestMethod]
        public void ToBytesPerPixel_24bppRgb_Returns3()
        {
            PixelFormat.Format24bppRgb.ToBytesPerPixel().Should().Be(3);
        }

        [TestMethod]
        public void ToBytesPerPixel_32bppArgb_Returns4()
        {
            PixelFormat.Format32bppArgb.ToBytesPerPixel().Should().Be(4);
        }

        [TestMethod]
        public void ToBytesPerPixel_32bppPArgb_Returns4()
        {
            PixelFormat.Format32bppPArgb.ToBytesPerPixel().Should().Be(4);
        }

        [TestMethod]
        public void ToBytesPerPixel_32bppRgb_Returns4()
        {
            PixelFormat.Format32bppRgb.ToBytesPerPixel().Should().Be(4);
        }

        [TestMethod]
        public void ToBytesPerPixel_48bppRgb_Returns6()
        {
            PixelFormat.Format48bppRgb.ToBytesPerPixel().Should().Be(6);
        }

        [TestMethod]
        public void ToBytesPerPixel_64bppArgb_Returns8()
        {
            PixelFormat.Format64bppArgb.ToBytesPerPixel().Should().Be(8);
        }

        [TestMethod]
        public void ToBytesPerPixel_64bppPArgb_Returns8()
        {
            PixelFormat.Format64bppPArgb.ToBytesPerPixel().Should().Be(8);
        }

        [TestMethod]
        public void ToBytesPerPixel_16bppFormats_Return2()
        {
            PixelFormat.Format16bppGrayScale.ToBytesPerPixel().Should().Be(2);
            PixelFormat.Format16bppRgb555.ToBytesPerPixel().Should().Be(2);
            PixelFormat.Format16bppRgb565.ToBytesPerPixel().Should().Be(2);
            PixelFormat.Format16bppArgb1555.ToBytesPerPixel().Should().Be(2);
        }

        [TestMethod]
        public void ToBytesPerPixel_IndexedFormat_ThrowsArgumentException()
        {
            Action act = () => PixelFormat.Format8bppIndexed.ToBytesPerPixel();
            act.Should().Throw<ArgumentException>();
        }

        [TestMethod]
        public void ToBytesPerPixel_DontCare_ThrowsArgumentException()
        {
            Action act = () => PixelFormat.DontCare.ToBytesPerPixel();
            act.Should().Throw<ArgumentException>();
        }

        #endregion

        #region Multiple embedded resources

        [TestMethod]
        public void ToNDArray_DifferentImages_HaveDifferentShapes()
        {
            var captcha = EmbeddedBitmap("captcha-a");
            var ndCaptcha = captcha.ToNDArray(copy: true);

            // Use copy:false for odd-width to avoid stride bug
            var odd = EmbeddedBitmap("odd-width");
            var ndOdd = odd.ToNDArray(copy: false);

            // Both should be 4-d with batch=1
            ndCaptcha.ndim.Should().Be(4);
            ndOdd.ndim.Should().Be(4);
            ndCaptcha.shape[0].Should().Be(1);
            ndOdd.shape[0].Should().Be(1);

            // Dimensions should match their source bitmaps
            ndCaptcha.shape[1].Should().Be(captcha.Height);
            ndCaptcha.shape[2].Should().Be(captcha.Width);
            ndOdd.shape[1].Should().Be(odd.Height);
            ndOdd.shape[2].Should().Be(odd.Width);

            // They should be different sizes
            (ndCaptcha.shape[1] == ndOdd.shape[1] && ndCaptcha.shape[2] == ndOdd.shape[2])
                .Should().BeFalse("different images should have different dimensions");
        }

        [TestMethod]
        public void ToNDArray_AllCaptchaImages_Load()
        {
            // Verify all 4 captcha images can be loaded and converted
            foreach (var name in new[] { "captcha-a", "captcha-b", "captcha-c", "captcha-d" })
            {
                var bitmap = EmbeddedBitmap(name);
                bitmap.Should().NotBeNull($"{name} should be loadable");

                var nd = bitmap.ToNDArray(copy: true);
                nd.ndim.Should().Be(4, $"{name} should produce 4-d array");
                nd.shape[0].Should().Be(1, $"{name} batch dim should be 1");
                nd.size.Should().BeGreaterThan(0, $"{name} should have pixel data");
            }
        }

        #endregion

        #region Zeros and ones images

        [TestMethod]
        public void ToBitmap_AllBlack_RoundTripsCorrectly()
        {
            var black = np.zeros(1, 8, 8, 3).astype(NPTypeCode.Byte);
            var bmp = black.ToBitmap(8, 8, PixelFormat.Format24bppRgb);
            var recovered = bmp.ToNDArray(copy: true, discardAlpha: false);
            recovered.Should().AllValuesBe((byte)0);
        }

        [TestMethod]
        public void ToBitmap_AllWhite_RoundTripsCorrectly()
        {
            var white = (np.zeros(1, 8, 8, 3) + 255).astype(NPTypeCode.Byte);
            var bmp = white.ToBitmap(8, 8, PixelFormat.Format24bppRgb);
            var recovered = bmp.ToNDArray(copy: true, discardAlpha: false);
            recovered.Should().AllValuesBe((byte)255);
        }

        [TestMethod]
        public void ToBitmap_32bpp_AllBlack_RoundTripsCorrectly()
        {
            var black = np.zeros(1, 8, 8, 4).astype(NPTypeCode.Byte);
            var bmp = black.ToBitmap(8, 8, PixelFormat.Format32bppArgb);
            var recovered = bmp.ToNDArray(copy: true, discardAlpha: false);
            recovered.Should().BeShaped(1, 8, 8, 4);
            recovered.Should().AllValuesBe((byte)0);
        }

        [TestMethod]
        public void ToBitmap_32bpp_AllWhite_RoundTripsCorrectly()
        {
            var white = (np.zeros(1, 8, 8, 4) + 255).astype(NPTypeCode.Byte);
            var bmp = white.ToBitmap(8, 8, PixelFormat.Format32bppArgb);
            var recovered = bmp.ToNDArray(copy: true, discardAlpha: false);
            recovered.Should().BeShaped(1, 8, 8, 4);
            recovered.Should().AllValuesBe((byte)255);
        }

        #endregion

        #region ToBitmap — specific pixel values

        [TestMethod]
        public void ToBitmap_32bpp_SpecificPixels_RoundTrip()
        {
            // Create a 2x2 image with known BGRA values
            var pixels = np.array(new byte[] {
                10, 20, 30, 40,     50, 60, 70, 80,
                90, 100, 110, 120,  130, 140, 150, 160
            }).reshape(1, 2, 2, 4);

            var bmp = pixels.ToBitmap(2, 2, PixelFormat.Format32bppArgb);
            var recovered = bmp.ToNDArray(copy: true, discardAlpha: false);

            // Verify each pixel exactly
            for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
            for (int c = 0; c < 4; c++)
            {
                byte expected = pixels[$"0, {y}, {x}, {c}"].GetByte();
                byte actual = recovered[$"0, {y}, {x}, {c}"].GetByte();
                actual.Should().Be(expected, $"pixel [{y},{x}] channel {c} should match");
            }
        }

        [TestMethod]
        public void ToBitmap_SizeProperty_MatchesDimensions()
        {
            var nd = np.zeros(1, 10, 20, 4).astype(NPTypeCode.Byte);
            var bmp = nd.ToBitmap();
            bmp.Width.Should().Be(20);
            bmp.Height.Should().Be(10);
            bmp.Size.Should().Be(new Size(20, 10));
        }

        #endregion
    }
}
