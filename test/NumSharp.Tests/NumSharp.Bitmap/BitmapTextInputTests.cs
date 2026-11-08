using System;
using System.Drawing;
using System.Drawing.Imaging;
using NumSharp.Tests.Utilities;

namespace NumSharp.Tests
{
    /// <summary>
    ///     Text into NumSharp.Bitmap: an image's pixels are bytes, and <c>ToBitmap</c> has no dtype check of its own — it
    ///     copies the array into the locked bitmap's <c>uint8</c> rows, padding a row first (<c>np.concatenate</c> with a
    ///     <c>uint8</c> zero column) when the format's stride is wider than the pixels. So a text array is refused where
    ///     any non-<c>uint8</c> array is: by the copy (<see cref="InvalidCastException"/>, the same text an int32 array
    ///     gets), or — for a <c>StringDType</c> array whose rows need padding — by the padding concatenate, with numpy's
    ///     own <c>DTypePromotionError</c> (numpy refuses <c>StringDType</c> + <c>uint8</c>, and promotes <c>U1</c> /
    ///     <c>S1</c> + <c>uint8</c> to <c>U3</c> / <c>S3</c>, which is why fixed-width text reaches the copy).
    /// </summary>
    /// <remarks>
    ///     The pixel source of every case is the 2 × 2 RGB image of digits <c>1</c>..<c>9</c>: text that numpy's
    ///     <c>astype(uint8)</c> WOULD parse, so a refusal here is the bridge's dtype rule, not a parse failure. The other
    ///     direction cannot meet text at all: <c>ToNDArray</c> / <c>AsNDArray</c> build <c>uint8</c> arrays only.
    /// </remarks>
    [WindowsOnly]
    [TestClass]
    public class BitmapTextInputTests : TestClass
    {
        /// <summary>The refusal of the copy into the bitmap's bytes for any dtype that is not uint8.</summary>
        private const string CopyRefusal = "Unable to perform CopyTo when T does not match dtype, use non-generic overload instead.";

        /// <summary>The 12 digit texts of a 2 × 2 RGB image.</summary>
        private static readonly string[] Digits = { "1", "2", "3", "4", "5", "6", "7", "8", "9", "1", "2", "3" };

        /// <summary>
        ///     Fixed-width text (<c>U1</c>, <c>U5</c>, <c>S1</c>; C order and a view with a reversed axis) into a 24-bit and a
        ///     32-bit image is refused by the copy with the SAME exception and text an int32 array gets — text is one more
        ///     dtype that is not <c>uint8</c>, nothing is drawn, and the padding concatenate before it promotes the text
        ///     exactly as numpy's <c>np.concatenate</c> does.
        /// </summary>
        [TestMethod]
        public void ToBitmap_FixedWidthText_IsRefusedLikeEveryNonByteDtype()
        {
            var inputs = new (string Name, NDArray Pixels)[]
            {
                ("U1 C", np.array(Digits).reshape(1, 2, 2, 3)),
                ("U5 C", np.array(Digits).astype(np.dtype("U5")).reshape(1, 2, 2, 3)),
                ("S1 C", np.array(Digits, np.dtype("S1")).reshape(1, 2, 2, 3)),
                ("U1 reversed columns", np.array(Digits).reshape(1, 2, 2, 3)[":, :, ::-1, :"]),
                ("int32 C (the reference)", np.arange(12).reshape(1, 2, 2, 3)),
            };
            foreach (var (name, pixels) in inputs)
            {
                new Action(() => pixels.ToBitmap(2, 2, PixelFormat.Format24bppRgb).Dispose())
                    .Should().Throw<InvalidCastException>($"{name}, 24-bit").WithMessage(CopyRefusal);
            }

            // 32-bit rows need no padding: the copy is reached directly.
            using NDArray argb = np.array(new[] { "1", "2", "3", "4", "5", "6", "7", "8", "9", "1", "2", "3", "4", "5", "6", "7" }).reshape(1, 2, 2, 4);
            new Action(() => argb.ToBitmap(2, 2, PixelFormat.Format32bppArgb).Dispose())
                .Should().Throw<InvalidCastException>().WithMessage(CopyRefusal);

            // The padding step is numpy's concatenate: U1 + uint8 promotes to U3 (numpy 2.4.2: '<U3'), S1 + uint8 to S3.
            using NDArray padded = np.concatenate((np.array(Digits).reshape(1, 2, 2, 3), np.zeros(new Shape(1, 2, 1, 3)).astype(NPTypeCode.Byte)), 2);
            padded.dtype.str.Should().Be("<U3");
            using NDArray paddedBytes = np.concatenate((np.array(Digits, np.dtype("S1")).reshape(1, 2, 2, 3), np.zeros(new Shape(1, 2, 1, 3)).astype(NPTypeCode.Byte)), 2);
            paddedBytes.dtype.str.Should().Be("|S3");
        }

        /// <summary>
        ///     A <c>StringDType</c> array whose rows need padding (a 24-bit image two pixels wide: a 6-byte row in an 8-byte
        ///     stride) is refused by the padding concatenate with numpy's own <c>DTypePromotionError</c> text for
        ///     <c>np.concatenate((StringDType, uint8))</c>; with no padding (32-bit rows) the copy refuses it like any
        ///     non-<c>uint8</c> dtype.
        /// </summary>
        [TestMethod]
        public void ToBitmap_StringDType_PaddedRowsRaiseNumpysPromotionError_UnpaddedRowsTheCopyRefusal()
        {
            using NDArray text = np.array(Digits, np.dtypes.StringDType()).reshape(1, 2, 2, 3);
            new Action(() => text.ToBitmap(2, 2, PixelFormat.Format24bppRgb).Dispose())
                .Should().Throw<DTypePromotionError>()
                .WithMessage("The DType <class 'numpy.dtypes.StringDType'> could not be promoted by <class 'numpy.dtypes.UInt8DType'>. *");

            using NDArray argb = np.array(new[] { "1", "2", "3", "4", "5", "6", "7", "8", "9", "1", "2", "3", "4", "5", "6", "7" },
                np.dtypes.StringDType()).reshape(1, 2, 2, 4);
            new Action(() => argb.ToBitmap(2, 2, PixelFormat.Format32bppArgb).Dispose())
                .Should().Throw<InvalidCastException>().WithMessage(CopyRefusal);
        }

        /// <summary>
        ///     The image → NDArray direction cannot produce text: a bitmap reads back as a <c>uint8</c> array (copied or
        ///     viewed), whatever its pixel format.
        /// </summary>
        [TestMethod]
        public void ToNDArray_IsAlwaysBytes()
        {
            using var bmp = new Bitmap(2, 2, PixelFormat.Format24bppRgb);
            using NDArray copy = bmp.ToNDArray(copy: true);
            copy.typecode.Should().Be(NPTypeCode.Byte);
            using NDArray view = bmp.ToNDArray(copy: false);
            view.typecode.Should().Be(NPTypeCode.Byte);
        }
    }
}
